#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""One entry point for a first run of the XC2 NS2 recompilation.

    python xc2_setup.py --repo <checkout> --install <folder> --nup <dump>.nup [--dry-run]

Steps, each skipped when its artifact is already there:
    doctor   tools/setup/doctor.ps1 -Json        what the machine is missing
    sources  nxrecomp tarball (deps.lock.json) + fetch_third_party.ps1
    runtime  the two cmake configures and builds (BUILD.md 4.1)
    lift     the seven modules to .lib (the repo's tools/build/build.ps1 does the same)
    link     the boot exe (repo: tools/build/build-boot.ps1)
    ingest   a .nup, or a folder that already holds the modules
    done     setup_state.tsv, exefs/ and romfs/ under --install

State lives in <install>/setup/status.json (the launcher reads it) and setup/setup.log.
Nothing is written into the dump; the romfs is linked, never copied.
"""
import argparse
import collections
import hashlib
import json
import os
import re
import shutil
import struct
import subprocess
import sys
import tarfile
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

MODULES = ["main", "sdk", "subsdk0", "subsdk1", "subsdk2", "subsdk3", "rtld"]
STATE_FILE = "status.json"
LOG_FILE = "setup.log"


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _redact_home(text):
    home = os.path.expanduser("~")
    return text.replace(home, "%USERPROFILE%") if home and home in text else text


class Runner:
    def __init__(self, install, dry):
        self.install = install
        self.state_dir = install / "setup"
        self.dry = dry
        self.log_path = self.state_dir / LOG_FILE
        self.status_path = self.state_dir / STATE_FILE
        self.status = {"state": "idle", "step": "", "note": "", "started_at": now(), "steps": []}
        self.live = {"detail": "", "percent": None}
        self.last_write = 0.0
        # Steps finished and steps in the page's list: the launcher paints its bar from "@progress" lines, and without
        # them it had nothing to paint (an empty bar that never moved, measured 2026-11-18).
        self.done_steps = 0
        self.total_steps = 0
        if not dry:
            self.state_dir.mkdir(parents=True, exist_ok=True)

    def log(self, line):
        stamp = datetime.now().strftime("%H:%M:%S")
        # The log gets pasted into issue reports: the home folder (an account name) is not something to hand out.
        text = "[%s] %s" % (stamp, _redact_home(line))
        print(text, flush=True)
        if not self.dry:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(text + "\n")

    def write_status(self, **kw):
        self.status.update(kw)
        self.status["updated_at"] = now()
        if not self.dry:
            tmp = self.status_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(self.status, indent=1), encoding="utf-8")
            os.replace(tmp, self.status_path)

    def announce(self):
        """One "@progress setup <done> <total>" line: the launcher's bar, and nothing else reads it."""
        print("@progress setup %d %d" % (self.done_steps, self.total_steps), flush=True)

    def step(self, name):
        self.live = {"detail": "", "percent": None}
        self.last_write = 0.0
        self.write_status(state="running", step=name, note="", progress=None)
        self.log("@step " + name)
        self.log("STEP: %s" % name)
        self.announce()
        return time.time()

    def end_step(self, name, started, ok, note=""):
        seconds = round(time.time() - started, 1)
        row = next((s for s in self.status["steps"] if s["name"] == name), None)
        if row is None:
            row = {"name": name}
            self.status["steps"].append(row)
        row.update(state="done" if ok else "failed", seconds=seconds, note=note)
        self.live = {"detail": "", "percent": None}
        if ok:
            self.done_steps += 1
        self.write_status(note=note, progress=None)
        self.log("STEP: %s %s %ss %s" % (name, "done" if ok else "FAILED", seconds, note))
        self.announce()
        return ok

    def note_progress(self, detail, percent=None):
        """Say what the current step is doing, at most once a second.

        A build step runs for minutes and its completion is the only other signal, so a front end polling
        status.json had nothing to show in between - measured 223.6 s of runtime with the file frozen at
        "runtime". The detail also goes into the step's own note, which is the field the checklist renders.
        """
        self.live = {"detail": detail, "percent": percent}
        t = time.time()
        if t - self.last_write < 1.0:
            return
        self.last_write = t
        row = next((s for s in self.status["steps"] if s["name"] == self.status["step"]), None)
        if row is None:
            # step() does not create the row - end_step() does - so the first progress line has to make it,
            # otherwise the detail has nowhere to go and the checklist still shows a blank step.
            row = {"name": self.status["step"], "state": "running", "seconds": 0, "note": ""}
            self.status["steps"].append(row)
        row["note"] = detail
        self.write_status(progress=dict(self.live))

    def run(self, argv, cwd=None, env=None, on_line=None):
        shown = " ".join(str(a) for a in argv)
        self.log("  $ " + shown)
        if self.dry:
            return 0
        full_env = os.environ.copy()
        if env:
            full_env.update(env)
        with open(self.log_path, "a", encoding="utf-8") as logf:
            logf.write("  $ %s\n" % shown)
            logf.flush()
            # Streamed rather than captured: the log used to appear only when the command exited, so a four
            # minute build looked like a hang to anyone watching setup.log.
            try:
                p = subprocess.Popen([str(a) for a in argv], cwd=str(cwd) if cwd else None, env=full_env,
                                     stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                     text=True, errors="replace", bufsize=1)
            except OSError as exc:
                msg = "  failed to start %s: %s" % (argv[0], exc)
                self.log(msg)
                logf.write(msg + "\n")
                logf.flush()
                return 127
            for line in p.stdout:
                logf.write("    " + line)
                logf.flush()
                if on_line is not None:
                    on_line(line.rstrip("\r\n"))
            return p.wait()


def first_file(*candidates):
    """The first path that exists.

    A checkout keeps the build and setup tools under tools/build and tools/setup; a release package from 0.1.3 keeps
    the old names (tools/bootstrap, tools/shader-cache). One engine runs in both, so it asks which one is there.
    """
    for c in candidates:
        if Path(c).is_file():
            return Path(c)
    return Path(candidates[0])


def find_powershell():
    """Return pwsh when installed, else fall back to built-in Windows PowerShell 5.1 (#58)."""
    candidates = [shutil.which("pwsh")]
    if os.name == "nt":
        sysroot = os.environ.get("SystemRoot") or r"C:\Windows"
        candidates += [
            r"C:\Program Files\PowerShell\7\pwsh.exe",
            shutil.which("powershell"),
            os.path.join(sysroot, "System32", "WindowsPowerShell", "v1.0", "powershell.exe"),
        ]
    for c in candidates:
        if c and Path(c).is_file():
            return str(c)
    return "powershell.exe" if os.name == "nt" else "pwsh"


def pwsh(*args):
    return [find_powershell(), "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", *args]


NINJA_LINE = re.compile(r"^\[(\d+)/(\d+)\]")


def build_progress(r, label):
    """Turn ninja's [done/total] into a percentage, and keep the last other line as the detail.

    cmake drives ninja, so every build step speaks this format; the counter is the only progress a user can
    act on, and without it the status file said "runtime" for four minutes and nothing else.
    """
    def on_line(line):
        s = line.strip()
        m = NINJA_LINE.match(s)
        if m:
            done, total = int(m.group(1)), int(m.group(2))
            r.note_progress("%s %d/%d" % (label, done, total), (done / total) if total else None)
        elif s:
            r.note_progress("%s: %s" % (label, s[:140]))
    return on_line


def import_toolchain(r):
    """Put the toolchain on PATH when XC2_ENV names the script that does it.

    The toolchain lives in a directory that script adds to PATH, so a machine that HAS cmake reports it missing
    unless the script ran first - and the engine then dies on a bare WinError 2 that says nothing useful. This is
    the same failure doctor.ps1 was fixed for; measured 2026-11-17 from a shell that never sourced it.
    """
    script = os.environ.get("XC2_ENV")
    if not script:
        return
    if not Path(script).is_file():
        r.log("  XC2_ENV points at %s, which is not a file; leaving PATH alone" % script)
        return
    try:
        p = subprocess.run([find_powershell(), "-NoProfile", "-Command",
                            ". $env:XC2_ENV *> $null; Write-Output $env:PATH"],
                           capture_output=True, text=True, timeout=180)
    except OSError as exc:
        r.log("  could not run PowerShell to read XC2_ENV: %s" % exc)
        return
    lines = [ln for ln in (p.stdout or "").splitlines() if ln.strip()]
    if p.returncode != 0 or not lines:
        r.log("  XC2_ENV did not yield a PATH (%s)" % (p.stderr or "").strip()[:200])
        return
    os.environ["PATH"] = lines[-1].strip()
    r.log("  toolchain on PATH, taken from %s" % script)


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def nso_ok(path):
    try:
        with open(path, "rb") as f:
            return f.read(4) == b"NSO0"
    except OSError:
        return False


def have_modules(folder):
    return all((folder / m).is_file() and nso_ok(folder / m) for m in MODULES)


def step_doctor(r, repo):
    started = r.step("doctor")
    rc = r.run(pwsh(first_file(repo / "tools" / "setup" / "doctor.ps1", repo / "tools" / "bootstrap" / "doctor.ps1"), "-Json"))
    return r.end_step("doctor", started, rc == 0, "the machine is ready" if rc == 0 else "missing tools, see the log")


def have_nxrecomp(dest):
    return (dest / "recomp").is_dir() and any(dest.rglob("*.py"))


def fetch_nxrecomp(r, repo, spec, dest):
    cache = repo / "work" / "setup" / spec["url"].rsplit("/", 1)[-1]
    cache.parent.mkdir(parents=True, exist_ok=True)
    if not cache.is_file() or sha256_of(cache) != spec["sha256"]:
        r.log("  downloading %s" % spec["url"])
        if r.dry:
            return 0
        with urllib.request.urlopen(spec["url"], timeout=120) as resp, open(cache, "wb") as f:
            shutil.copyfileobj(resp, f)
        got = sha256_of(cache)
        if got != spec["sha256"]:
            r.log("  sha256 mismatch: %s != %s" % (got, spec["sha256"]))
            return 1
    r.log("  unpacking into %s" % dest)
    if r.dry:
        return 0
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(cache, "r:gz") as tf:
        names = tf.getnames()
        top = names[0].split("/")[0] if names else ""
        strip = 1 if top and all(n == top or n.startswith(top + "/") for n in names) else 0
        for member in tf.getmembers():
            parts = member.name.split("/")[strip:]
            name = "/".join(p for p in parts if p)
            if not name:
                continue
            target = dest / name
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
            elif member.isfile():
                target.parent.mkdir(parents=True, exist_ok=True)
                with tf.extractfile(member) as src, open(target, "wb") as out:
                    shutil.copyfileobj(src, out)
    return 0


def step_sources(r, repo):
    started = r.step("sources")
    lock = json.loads(first_file(repo / "tools" / "setup" / "deps.lock.json", repo / "tools" / "bootstrap" / "deps.lock.json").read_text(encoding="utf-8"))
    nx = next((s for s in lock["sources"] if s["name"] == "nxrecomp"), None)
    dest = repo / (nx["dest"] if nx and nx.get("dest") else "third_party/nxrecomp")
    if have_nxrecomp(dest):
        r.log("  nxrecomp already in place")
    elif nx is None:
        return r.end_step("sources", started, False, "deps.lock.json has no nxrecomp entry")
    elif fetch_nxrecomp(r, repo, nx, dest) != 0:
        return r.end_step("sources", started, False, "the recompiler could not be fetched")
    textures = repo / "third_party" / "fefw-recomp" / "tools" / "textures" / "third_party" / "astc-encoder"
    if textures.is_dir():
        r.log("  texture sources already fetched")
    elif r.run(pwsh(repo / "third_party" / "fefw-recomp" / "tools" / "textures" / "scripts" / "fetch_third_party.ps1"), cwd=repo) != 0:
        return r.end_step("sources", started, False, "fetch_third_party.ps1 failed")
    return r.end_step("sources", started, True)


def step_runtime(r, repo):
    started = r.step("runtime")
    runtime = repo / "third_party" / "fefw-recomp" / "runtime"
    gfx = repo / "third_party" / "fefw-recomp" / "build" / "runtime-gfx"
    game = repo / "third_party" / "fefw-recomp" / "build" / "xc2-game-probe"
    cc = ["-DCMAKE_C_COMPILER=clang-cl", "-DCMAKE_CXX_COMPILER=clang-cl"]
    if not (gfx / "build.ninja").is_file():
        if r.run(["cmake", "-S", runtime, "-B", gfx, "-G", "Ninja", "-DCMAKE_BUILD_TYPE=Release",
                  "-DNXRT_BUILD_GFX=ON", "-DNXRT_BUILD_GAMES=OFF", "-DNXRT_BUILD_TESTS=OFF", *cc]) != 0:
            return r.end_step("runtime", started, False, "the graphics shell could not be configured")
    if r.run(["cmake", "--build", gfx, "--target", "nxrt", "nxrt_gfx"], on_line=build_progress(r, "runtime")) != 0:
        return r.end_step("runtime", started, False, "nxrt/nxrt_gfx did not build")
    if not (game / "build.ninja").is_file():
        if r.run(["cmake", "-S", runtime, "-B", game, "-G", "Ninja", "-DCMAKE_BUILD_TYPE=Release",
                  "-DNXRT_BUILD_GFX=ON", "-DNXRT_BUILD_GAMES=ON", "-DNXRT_GAMES=xc2ns2",
                  "-DNXRT_TEST_GAME=xc2ns2", "-DNXRT_BUILD_TESTS=OFF", *cc]) != 0:
            return r.end_step("runtime", started, False, "the game module could not be configured")
    if r.run(["cmake", "--build", game, "--target", "nxrt_game_xc2ns2_core", "nxrt_game_xc2ns2_data"],
              on_line=build_progress(r, "runtime")) != 0:
        return r.end_step("runtime", started, False, "nxrt_game_xc2ns2_* did not build")
    return r.end_step("runtime", started, True)


def prepare_patched_exefs(r, repo, install):
    """M3: apply active .pchtxt/.ips mod code patches onto a staging copy (never modifying vanilla exefs).

    Invalidates only modules whose patch state changed since the previous build so Set up incrementally
    re-lifts and re-links without rebuilding unaffected modules. Returns the ExeFS Path to lift/run with.
    """
    vanilla = install / "exefs"
    mods_dir = install / "mods"
    patch_mod = repo / "tools" / "mods" / "exefs_patch.py"
    patched_dir = install / "build" / "exefs-patched"
    applied_file = install / "build" / "m3_applied.json"
    if not vanilla.is_dir() or not mods_dir.is_dir() or not patch_mod.is_file():
        return vanilla

    import importlib.util
    spec = importlib.util.spec_from_file_location("xc2_exefs_patch", patch_mod)
    if spec is None or spec.loader is None:
        return vanilla
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    ini_path = install / "nxrt.ini"
    state = mod.apply_exefs_patches(vanilla, patched_dir, mods_dir, ini_path=ini_path if ini_path.is_file() else None, dry_run=r.dry)
    prev = {}
    if applied_file.is_file():
        try:
            prev = json.loads(applied_file.read_text(encoding="utf-8")).get("modules", {})
        except Exception:
            prev = {}

    changed = [m for m, h in state.get("modules", {}).items() if prev.get(m, "") != h]
    pending_file = install / "build" / "m3_pending.json"
    if changed and not r.dry:
        r.log("  m3: code patches changed for: %s" % ", ".join(changed))
        for m in changed:
            for stale_dir in (
                install / "build" / "lift" / m,
                install / "build" / "obj-zig" / m,
                repo / "xc2" / "build" / "301" / "lift" / m,
                repo / "xc2" / "build" / "301" / "obj" / m,
            ):
                shutil.rmtree(stale_dir, ignore_errors=True)
            for stale_file in (
                repo / "xc2" / "build" / "301" / "lib" / (m + ".lib"),
            ):
                try:
                    stale_file.unlink()
                except FileNotFoundError:
                    pass
        for exe_cand in (install / "xc2_boot.exe", repo / "xc2" / "build" / "301" / "boot" / "xc2_boot.exe"):
            try:
                exe_cand.unlink()
            except FileNotFoundError:
                pass
        pending_file.parent.mkdir(parents=True, exist_ok=True)
        pending_file.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
    elif not changed and not applied_file.is_file() and not r.dry:
        pending_file.parent.mkdir(parents=True, exist_ok=True)
        pending_file.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")

    for entry in state.get("applied", []):
        r.log("  m3: %s <- %s (%d record(s))" % (entry["module"], entry["source"], entry["records"]))
    for entry in state.get("unmatched", []):
        r.log("  m3: skipped %s (build_id %s does not match this dump)" % (entry["source"], entry["build_id"]))
    return patched_dir if state.get("active") else vanilla


def finalize_m3_state(r, install):
    """Promote build/m3_pending.json -> m3_applied.json and clear [mods] pending_rebuild after link succeeds."""
    if r.dry:
        return
    pending = install / "build" / "m3_pending.json"
    applied = install / "build" / "m3_applied.json"
    if pending.is_file():
        applied.parent.mkdir(parents=True, exist_ok=True)
        applied.write_text(pending.read_text(encoding="utf-8"), encoding="utf-8")
        try:
            pending.unlink()
        except OSError:
            pass
    ini_path = install / "nxrt.ini"
    if ini_path.is_file():
        lines = ini_path.read_text(encoding="utf-8", errors="replace").splitlines(True)
        in_mods = False
        changed = False
        out = []
        for ln in lines:
            st = ln.strip()
            if st.startswith("[") and st.endswith("]"):
                in_mods = st[1:-1].strip().lower() == "mods"
            elif in_mods and "=" in ln and not st.startswith(("#", ";")):
                k, _ = ln.split("=", 1)
                if k.strip().lower() == "pending_rebuild" and ln.split("=", 1)[1].strip():
                    eol = "\r\n" if ln.endswith("\r\n") else ("\n" if ln.endswith("\n") else "")
                    ln = f"{k.rstrip()} ={eol}"
                    changed = True
            out.append(ln)
        if changed:
            ini_path.write_text("".join(out), encoding="utf-8")


def code_patch_mods(install):
    """The mods whose code patches are compiled into this install's xc2_boot.exe, or [] when there are none.

    A wrong code patch - a Switch 1 cheat whose addresses were never ported, which is the common case - yields a
    binary that dies on start-up. The headless prefill is the first thing to run that binary, so the failure lands on
    the shaders step and reads like a graphics fault. Naming the mods is what turns it into something a player can
    act on (issue #75).
    """
    state = Path(install) / "build" / "exefs-patched" / "m3_state.json"
    if not state.is_file():
        return []
    try:
        applied = json.loads(state.read_text(encoding="utf-8")).get("applied") or []
    except (OSError, ValueError):
        return []
    mods = []
    for entry in applied:
        name = str(entry.get("mod") or str(entry.get("source", "")).split(":", 1)[0]).strip()
        if name and name not in mods:
            mods.append(name)
    return mods


def step_lift(r, repo, install, jobs=0):
    started = r.step("lift")
    exefs_dir = prepare_patched_exefs(r, repo, install)
    lib = repo / "xc2" / "build" / "301" / "lib"
    if all((lib / (m + ".lib")).is_file() for m in MODULES):
        return r.end_step("lift", started, True, "the libraries are already there")
    argv = pwsh(repo / "tools" / "build" / "build.ps1", "-Exefs", exefs_dir, "-Run:$false",
                "-Modules", ",".join(MODULES), "-LiftDir", repo / "xc2" / "build" / "301" / "lift",
                "-ObjDir", repo / "xc2" / "build" / "301" / "obj", "-LibDir", lib)
    if jobs > 0:
        # The build script's own default is 10 concurrent clang-cl, which is what it takes to run the largest
        # translation units out of memory on a busy machine: measured 10 failures that each compiled in 2-4 s by
        # hand. A front end that knows the machine can say otherwise.
        r.log("  jobs: %d" % jobs)
        argv += ["-Jobs", str(jobs)]
    rc = r.run(argv, cwd=repo, on_line=build_progress(r, "lift"))
    return r.end_step("lift", started, rc == 0, "" if rc == 0 else "build.ps1 failed (its stub link is expected to fail)")


def step_link(r, repo, install):
    started = r.step("link")
    exe = repo / "xc2" / "build" / "301" / "boot" / "xc2_boot.exe"
    lib = repo / "xc2" / "build" / "301" / "lib"
    # The exe is the lifted game code linked against the runtime headers, so it is stale as soon as one of those
    # libraries is newer. Checking only that the file exists kept an old exe across a runtime change, and the mixed
    # pair showed up as rendering faults that a fresh build fixed (a comment on the 0.2.1 video).
    newest_lib = max((p.stat().st_mtime for p in lib.glob("*.lib")), default=0.0)
    if exe.is_file() and exe.stat().st_mtime >= newest_lib:
        return r.end_step("link", started, True, "the boot exe is already there")
    patched = install / "build" / "exefs-patched"
    exefs_dir = patched if (patched / "m3_state.json").is_file() and (patched / "main").is_file() else install / "exefs"
    rc = r.run(pwsh(repo / "tools" / "build" / "build-boot.ps1", "-Exefs", exefs_dir,
                    "-OutDir", repo / "xc2" / "build" / "301" / "boot", "-LibDir", repo / "xc2" / "build" / "301" / "lib",
                    "-MainLib", repo / "xc2" / "build" / "301" / "lib" / "main.lib", "-Defines", "XC2_MODULES_7"), cwd=repo,
                on_line=build_progress(r, "link"))
    if rc == 0:
        finalize_m3_state(r, install)
    return r.end_step("link", started, rc == 0, "" if rc == 0 else "build-boot.ps1 failed")


# ---- package mode ---------------------------------------------------------------------------------------------
# A release package carries the runtime as a DLL and the two tools that compile the lifted code, so the same engine
# runs there with nothing installed: no cmake, no clang-cl, no XC2_ENV, no checkout. Whether this is one is decided
# by the folder itself, not by a flag, so a package cannot be mistaken for a checkout or the other way round.

# Guest load addresses, the table tools/build/build.ps1 lifts with.
BASES = {
    "main": "0x7100000000", "sdk": "0x7500000000", "subsdk0": "0x7600000000", "subsdk1": "0x7640000000",
    "subsdk2": "0x7680000000", "subsdk3": "0x76C0000000", "rtld": "0x7700000000",
}
# Larger guest-code bucket (0x40000 vs nxrecomp's default 0xC000) cuts ~537 translation units to ~120 balanced files
# without tripping --shards' imbalance on sparse modules such as sdk.
PACKAGE_BUCKET = 0x40000
ZIG_FLAGS = ["-target", "x86_64-windows-gnu", "-std=c++20", "-O1", "-w", "-fno-exceptions", "-fno-rtti",
             "-fno-strict-aliasing", "-ffp-contract=off", "-fno-math-errno", "-fno-threadsafe-statics",
             "-DNX_INLINE_INDIRECT"]
_CPU_VARIANTS = {}


def package_tools(repo):
    """(python, zig, xc2_rt.lib) when repo is a release package, else None."""
    py = repo / "tools" / "python" / "python.exe"
    zig = repo / "tools" / "build" / "zig" / "zig.exe"
    lib = repo / "runtime" / "xc2_rt.lib"
    if py.is_file() and zig.is_file() and lib.is_file():
        return py, zig, lib
    return None


def cpu_variant(zig):
    """The -mcpu this processor runs: '+fma' needs FMA3 (Haswell / Piledriver or newer), so an older one gets the
    compatible set instead of an illegal instruction. Same two variants as tools/build/build-release.ps1."""
    key = str(zig)
    if key in _CPU_VARIANTS:
        return _CPU_VARIANTS[key]
    try:
        out = subprocess.run([str(zig), "targets"], capture_output=True, text=True, errors="replace", timeout=120)
        text = out.stdout
        i = text.find(".native")
        j = text.find(".features", i)
        if i < 0 or j < 0:
            val = "x86_64+cx16+fma"
            _CPU_VARIANTS[key] = val
            return val
        feats = re.findall(r'"([^"]+)"', text[j:text.find("}", j)])
    except (OSError, subprocess.SubprocessError):
        val = "x86_64+cx16+fma"
        _CPU_VARIANTS[key] = val
        return val
    need = ("cx16", "sse4_2", "avx", "fma")
    val = "x86_64+cx16+fma" if all(f in feats for f in need) else "x86_64+cx16+sse4_2+popcnt"
    _CPU_VARIANTS[key] = val
    return val


def step_doctor_package(r, repo, tools):
    started = r.step("doctor")
    py, zig, lib = tools
    rc = r.run([str(py), "-c", "import capstone, lz4.block, zstandard, nxrecomp"])
    if rc != 0:
        return r.end_step("doctor", started, False, "the bundled Python cannot import the lifter (capstone/lz4/zstandard/nxrecomp)")
    r.log("  python %s" % py)
    r.log("  zig    %s (%s)" % (zig, cpu_variant(zig)))
    r.log("  runtime %s" % lib)
    val = repo / "tools" / "spirv-val" / "spirv-val.exe"
    if val.is_file():
        r.log("  spirv-val %s (optional; package setup skips per-shader validation)" % val)
    return r.end_step("doctor", started, True, "the package carries everything")


# ---- install manifest (content-hashed upgrade state) ----------------------------------------------------------
# One manifest at <install>/build/install-manifest.json tracks content hashes for lift, link, shaders, and
# prefill so upgrades only re-run the stages whose inputs actually changed (never trusting file mtimes alone).
# Existing installs without a manifest are seeded in-place on first upgrade without re-lifting or re-translating.
MANIFEST_VERSION = 1
PREFILL_SCHEMA_VERSION = 1


def manifest_path(install):
    return Path(install) / "build" / "install-manifest.json"


def load_install_manifest(install):
    p = manifest_path(install)
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(data, dict) and data.get("version") == MANIFEST_VERSION:
            return data
    except (OSError, ValueError):
        pass
    return {}


def save_install_manifest(install, data, dry=False):
    if dry:
        return
    p = manifest_path(install)
    p.parent.mkdir(parents=True, exist_ok=True)
    out = dict(data)
    out["version"] = MANIFEST_VERSION
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, p)


def _sha256_bytes(blob):
    return hashlib.sha256(blob).hexdigest()


def file_fingerprint_fast(path):
    """Content fingerprint that stays O(1) on multi-GB archives (size + first/last 64 KiB SHA-256)."""
    p = Path(path)
    try:
        st = p.stat()
    except OSError:
        return ""
    sz = st.st_size
    if sz <= 262144:
        return "%d:%s" % (sz, sha256_of(p))
    h = hashlib.sha256()
    with open(p, "rb") as f:
        h.update(f.read(65536))
        f.seek(max(0, sz - 65536))
        h.update(f.read(65536))
    return "%d:%s" % (sz, h.hexdigest())


def _hash_named_files(pairs):
    h = hashlib.sha256()
    for label, path in sorted(pairs, key=lambda kv: kv[0]):
        p = Path(path)
        if p.is_file():
            h.update(label.encode("utf-8"))
            h.update(b"\0")
            h.update(sha256_of(p).encode("ascii"))
            h.update(b"\n")
    return h.hexdigest()


def _pe_sections(data):
    if len(data) < 0x40 or data[:2] != b"MZ":
        return None
    pe = struct.unpack_from("<I", data, 0x3C)[0]
    if pe + 24 > len(data) or data[pe : pe + 4] != b"PE\0\0":
        return None
    nsec = struct.unpack_from("<H", data, pe + 6)[0]
    opt_sz = struct.unpack_from("<H", data, pe + 20)[0]
    opt_off = pe + 24
    if opt_off + opt_sz > len(data) or opt_sz < 120:
        return None
    magic = struct.unpack_from("<H", data, opt_off)[0]
    if magic not in (0x10B, 0x20B):
        return None
    dd_off = opt_off + (112 if magic == 0x20B else 96)
    sec_off = opt_off + opt_sz
    if sec_off + nsec * 40 > len(data):
        return None
    secs = []
    for i in range(nsec):
        o = sec_off + i * 40
        vsize, va, rsize, roff = struct.unpack_from("<IIII", data, o + 8)
        secs.append((va, max(vsize, rsize), roff))
    return magic, dd_off, secs


def _rva_to_off(secs, rva):
    for va, sz, roff in secs:
        if va <= rva < va + sz:
            return roff + (rva - va)
    return None


def pe_export_names(path):
    """Sorted exported symbol names of a PE DLL (e.g. xc2_rt.dll), or None if not a readable PE."""
    try:
        data = Path(path).read_bytes()
    except OSError:
        return None
    parsed = _pe_sections(data)
    if parsed is None:
        return None
    _, dd_off, secs = parsed
    if dd_off + 8 > len(data):
        return None
    exp_rva, _ = struct.unpack_from("<II", data, dd_off)
    if not exp_rva:
        return []
    eo = _rva_to_off(secs, exp_rva)
    if eo is None or eo + 40 > len(data):
        return None
    num_names = struct.unpack_from("<I", data, eo + 24)[0]
    names_rva = struct.unpack_from("<I", data, eo + 32)[0]
    no = _rva_to_off(secs, names_rva)
    if no is None or no + num_names * 4 > len(data):
        return None
    names = []
    for i in range(num_names):
        nrva = struct.unpack_from("<I", data, no + i * 4)[0]
        p = _rva_to_off(secs, nrva)
        if p is None or p >= len(data):
            return None
        end = data.find(b"\0", p)
        if end < 0:
            return None
        names.append(data[p:end].decode("ascii", "replace"))
    return sorted(names)


def pe_imported_symbols(path, dll_name="xc2_rt.dll"):
    """Sorted symbol names imported from `dll_name` by a PE exe, or None if not a readable PE."""
    try:
        data = Path(path).read_bytes()
    except OSError:
        return None
    parsed = _pe_sections(data)
    if parsed is None:
        return None
    magic, dd_off, secs = parsed
    if dd_off + 16 > len(data):
        return None
    imp_rva, _ = struct.unpack_from("<II", data, dd_off + 8)
    if not imp_rva:
        return []
    io = _rva_to_off(secs, imp_rva)
    if io is None:
        return None
    want = dll_name.lower()
    is_pe64 = magic == 0x20B
    entry_sz = 8 if is_pe64 else 4
    ord_mask = (1 << 63) if is_pe64 else (1 << 31)
    syms = []
    pos = io
    while pos + 20 <= len(data):
        ilt_rva, _, _, name_rva, iat_rva = struct.unpack_from("<IIIII", data, pos)
        if ilt_rva == 0 and name_rva == 0 and iat_rva == 0:
            break
        pos += 20
        np = _rva_to_off(secs, name_rva)
        if np is None:
            continue
        end = data.find(b"\0", np)
        if end < 0:
            continue
        mod = data[np:end].decode("ascii", "replace").lower()
        if mod != want:
            continue
        tbl_rva = ilt_rva or iat_rva
        tp = _rva_to_off(secs, tbl_rva)
        if tp is None:
            return None
        while tp + entry_sz <= len(data):
            val = struct.unpack_from("<Q" if is_pe64 else "<I", data, tp)[0]
            if val == 0:
                break
            tp += entry_sz
            if val & ord_mask:
                return None  # ordinal import cannot be checked by symbol name
            hp = _rva_to_off(secs, val & 0x7FFFFFFF)
            if hp is None or hp + 2 >= len(data):
                return None
            send = data.find(b"\0", hp + 2)
            if send < 0:
                return None
            syms.append(data[hp + 2 : send].decode("ascii", "replace"))
    return sorted(set(syms))


def parse_def_exports(path):
    """Sorted export names from a .def file (xc2/release/src/xc2_rt.def), or None if absent."""
    p = Path(path)
    if not p.is_file():
        return None
    in_exports = False
    names = []
    for raw in p.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.split(";", 1)[0].strip()
        if not line:
            continue
        if line.upper() == "EXPORTS":
            in_exports = True
            continue
        if in_exports:
            tok = line.split()[0]
            if tok:
                names.append(tok)
    return sorted(names) if names else None


def runtime_abi_signature(repo, install, rtlib=None):
    """ABI fingerprint of xc2_rt.dll based on its exported C symbol table (xc2_rt.def).

    Routine renderer/audio/UI updates keep the same 17 C exports, so xc2_boot.exe does not need to be
    relinked by Zig when only xc2_rt.dll internals change. Falls back to xc2_rt.def or rtlib SHA-256.
    """
    dll_cands = []
    if rtlib is not None:
        dll_cands.append(Path(rtlib).parent / "xc2_rt.dll")
    dll_cands += [
        Path(repo) / "runtime" / "xc2_rt.dll",
        Path(install) / "runtime" / "xc2_rt.dll",
        Path(install) / "xc2_rt.dll",
    ]
    for dll in dll_cands:
        if dll.is_file():
            exports = pe_export_names(dll)
            if exports:
                return {"source": "pe_exports", "count": len(exports), "sha256": _sha256_bytes("\n".join(exports).encode("ascii"))}
    for defp in (Path(repo) / "runtime" / "xc2_rt.def", Path(repo) / "xc2" / "release" / "src" / "xc2_rt.def"):
        exports = parse_def_exports(defp)
        if exports:
            return {"source": "def", "count": len(exports), "sha256": _sha256_bytes("\n".join(exports).encode("ascii"))}
    for fallback in ([Path(rtlib)] if rtlib is not None else []) + dll_cands:
        if fallback.is_file():
            return {"source": "file_sha256", "sha256": sha256_of(fallback)}
    return None


def exe_abi_compatible(exe, repo, install, rtlib=None):
    """True when an existing PE xc2_boot.exe imports xc2_rt.dll symbols that are all exported by the current DLL."""
    imports = pe_imported_symbols(exe, "xc2_rt.dll")
    if not imports:
        return False
    dll_cands = []
    if rtlib is not None:
        dll_cands.append(Path(rtlib).parent / "xc2_rt.dll")
    dll_cands += [Path(repo) / "runtime" / "xc2_rt.dll", Path(install) / "runtime" / "xc2_rt.dll"]
    for dll in dll_cands:
        if dll.is_file():
            exports = pe_export_names(dll)
            if exports:
                return set(imports).issubset(set(exports))
    return False


def lift_inputs_hash(repo, exefs_dir):
    """Per-module + lifter tool content hashes for step_lift_package."""
    repo = Path(repo)
    exefs_dir = Path(exefs_dir)
    nx_roots = [
        repo / "tools" / "python" / "Lib" / "site-packages" / "nxrecomp",
        repo / "third_party" / "nxrecomp" / "recomp" / "nxrecomp",
    ]
    nx_pairs = []
    for root in nx_roots:
        if root.is_dir():
            for f in root.rglob("*.py"):
                nx_pairs.append(("nxrecomp/" + f.relative_to(root).as_posix(), f))
            break
    h_tool = hashlib.sha256()
    h_tool.update(("bucket=0x%x\n" % PACKAGE_BUCKET).encode("ascii"))
    h_tool.update(_hash_named_files(nx_pairs).encode("ascii"))
    tool_hash = h_tool.hexdigest()

    sites = repo / "xc2" / "sites"
    mods = {}
    for m in MODULES:
        hm = hashlib.sha256()
        hm.update(file_fingerprint_fast(exefs_dir / m).encode("ascii"))
        hm.update(b"\n")
        sf = sites / ("sites-%s.txt" % m)
        if sf.is_file():
            hm.update(sha256_of(sf).encode("ascii"))
        mods[m] = hm.hexdigest()
    return {"tool": tool_hash, "modules": mods}


def link_inputs_hash(repo, install, rtlib, lift_state, cpu_var):
    """Content hash governing whether xc2_boot.exe must be relinked."""
    repo = Path(repo)
    abi = runtime_abi_signature(repo, install, rtlib)
    if abi is None:
        return None
    pairs = [
        ("xc2/src/xc2_stub.cpp", repo / "xc2" / "src" / "xc2_stub.cpp"),
        ("xc2/src/xc2_modules.h", repo / "xc2" / "src" / "xc2_modules.h"),
    ]
    payload = {
        "abi": abi,
        "cpu": cpu_var,
        "stub": _hash_named_files(pairs),
        "lift": lift_state,
    }
    return _sha256_bytes(json.dumps(payload, sort_keys=True).encode("utf-8"))


def shader_inputs_hash(repo, install):
    """Content hash covering bf2.ard, the shader translator (spirv_harness + Ryujinx DLLs), and cache scripts."""
    repo = Path(repo)
    install = Path(install)
    pairs = []
    hdir = repo / "tools" / "spirv_harness"
    if hdir.is_dir():
        # Include all top-level translator binaries/config (Spv.Generator.dll, AmpereDecoder.dll, Gommon.dll,
        # Ryujinx*.dll, spirv_harness.dll/.exe/.json) while ignoring debug-symbol .pdb files.
        for f in sorted(hdir.iterdir()):
            if not f.is_file():
                continue
            n = f.name.lower()
            if n.endswith(".pdb"):
                continue
            if n.endswith((".dll", ".exe", ".json")):
                pairs.append(("harness/" + f.name, f))
    for sdir in (repo / "tools" / "shader-cache", repo / "tools" / "shader" / "cache"):
        if sdir.is_dir():
            for f in sorted(sdir.glob("*.py")):
                pairs.append(("shader-cache/" + f.name, f))
            break
    patch_ds = repo / "tools" / "shader" / "patch_ds_samplers.py"
    if patch_ds.is_file():
        pairs.append(("shader/patch_ds_samplers.py", patch_ds))
    payload = {
        "ard": file_fingerprint_fast(install / "romfs" / "bf2.ard"),
        "tools": _hash_named_files(pairs),
    }
    return _sha256_bytes(json.dumps(payload, sort_keys=True).encode("utf-8"))


def prefill_inputs_hash(shader_hash):
    payload = {"schema": PREFILL_SCHEMA_VERSION, "shaders": shader_hash}
    return _sha256_bytes(json.dumps(payload, sort_keys=True).encode("utf-8"))


def drop_preload_pack(r, spirv):
    """Explicitly remove stale spirv.preload.pack after shader refresh (the pack wins over loose .spv files)."""
    spirv = Path(spirv)
    removed = []
    for cand in (spirv.parent / "spirv.preload.pack", spirv / "spirv.preload.pack"):
        if cand.is_file():
            if not r.dry:
                try:
                    cand.unlink()
                except OSError:
                    continue
            removed.append(cand.name)
    if removed:
        r.log("  shaders: dropped stale %s so the runtime rebuilds it from refreshed .spv files" % ", ".join(removed))
    return removed


def step_lift_package(r, repo, install, py, jobs):
    started = r.step("lift")
    exefs_dir = prepare_patched_exefs(r, repo, install)
    # Build products live with the game, not in the package: a player's install folder is theirs to fill, and the
    # package stays exactly as it was unpacked.
    lift = install / "build" / "lift"
    obj = install / "build" / "obj-zig"
    sites = repo / "xc2" / "sites"
    manifest = load_install_manifest(install)
    want_lift = lift_inputs_hash(repo, exefs_dir)
    prev_lift = manifest.get("lift")
    all_lifted = all((lift / m / ("module_%s.cpp" % m)).is_file() for m in MODULES)
    if prev_lift is None and all_lifted:
        # Legacy install seeding: do not force an existing install to re-lift on the first manifest-enabled upgrade.
        r.log("  manifest: seeded lift from existing build/lift (legacy install)")
        manifest["lift"] = want_lift
        save_install_manifest(install, manifest, dry=r.dry)
        return r.end_step("lift", started, True, "the lifted code is already there")
    if isinstance(prev_lift, dict) and prev_lift != want_lift:
        tool_changed = prev_lift.get("tool") != want_lift.get("tool")
        prev_mods = prev_lift.get("modules") if isinstance(prev_lift.get("modules"), dict) else {}
        stale_mods = [m for m in MODULES if tool_changed or prev_mods.get(m) != want_lift["modules"].get(m)]
        if stale_mods:
            r.log("  lift inputs changed for: %s" % ", ".join(stale_mods))
            if not r.dry:
                for m in stale_mods:
                    shutil.rmtree(lift / m, ignore_errors=True)
                    shutil.rmtree(obj / m, ignore_errors=True)
    n = jobs if jobs > 0 else max(2, (os.cpu_count() or 4) - 1)
    done = 0
    tasks = []
    for m in MODULES:
        out = lift / m
        if (out / ("module_%s.cpp" % m)).is_file():
            r.log("  %s: already lifted" % m)
            done += 1
            continue
        s = sites / ("sites-%s.txt" % m)
        tasks.append({
            "name": m,
            "input": str(exefs_dir / m),
            "out": str(out),
            "base": BASES[m],
            "sites": str(s) if s.is_file() else None,
            "jobs": n,
            "bucket": PACKAGE_BUCKET,
        })
    if tasks:
        for t in tasks:
            r.log("  lift %s (bucket 0x%x)" % (t["name"], t["bucket"]))
        if not r.dry:
            lift.mkdir(parents=True, exist_ok=True)
            jobs_file = lift / "_lift_jobs.json"
            jobs_file.write_text(json.dumps(tasks), encoding="utf-8")
            # Lift every pending module inside a single Python process so interpreter + capstone startup happens once.
            script = (
                "import json,sys; from nxrecomp.program import lift_module\n"
                "for j in json.load(open(sys.argv[1], encoding='utf-8')):\n"
                "  k = 'nso' if open(j['input'],'rb').read(4)==b'NSO0' else 'static_elf'\n"
                "  spec = {'kind':k,'path':j['input'],'base':int(j['base'],16),'name':j['name'],'game':'xc2ns2',"
                "'patches':None,'sites':j['sites'],'census':None,'counters':None,'fps_sites':None,'index':None}\n"
                "  lift_module(spec, j['out'], jobs=j['jobs'], bucket=j['bucket'])\n"
                "  print('@lift_done ' + j['name'], flush=True)\n"
            )

            def on_lift_line(line):
                nonlocal done
                if line.startswith("@lift_done "):
                    m = line.split(" ", 1)[1].strip()
                    done += 1
                    r.note_progress("lift %s (%d/%d)" % (m, done, len(MODULES)), 100.0 * done / len(MODULES))

            rc = r.run([str(py), "-u", "-c", script, str(jobs_file)], cwd=repo, on_line=on_lift_line)
            try:
                jobs_file.unlink()
            except OSError:
                pass
            if rc != 0:
                return r.end_step("lift", started, False, "the lifter failed")
    manifest["lift"] = want_lift
    save_install_manifest(install, manifest, dry=r.dry)
    return r.end_step("lift", started, True)


def copy_runtime_dlls(r, runtime_dir, install):
    """Every DLL the package put in runtime/ goes beside the exe: xc2_rt.dll, and the FFmpeg shared libraries the film's
    hardware decoders load. Nothing in the copy depends on the exe having just been built, so an install that keeps its
    exe still gets them - an in-place upgrade used to skip this copy and then fail to start with 0xC0000135
    (avutil-61.dll / avcodec-63.dll missing, issue #2). A runtime folder without FFmpeg is not an error: the runtime
    then says "built without FFmpeg" and decodes films on the CPU."""
    n = 0
    for dll in sorted(Path(runtime_dir).glob("*.dll")):
        shutil.copy2(str(dll), str(Path(install) / dll.name))
        n += 1
    r.log("  runtime DLLs: %d beside the exe" % n)
    return n


def clear_high_entropy_va(exe):
    """Disable IMAGE_DLLCHARACTERISTICS_HIGH_ENTROPY_VA (0x20) on xc2_boot.exe so Windows keeps bottom-up allocations
    low and out of the guest alias/module address ranges (issue #25). Zig's GNU linker omits /HIGHENTROPYVA:NO."""
    with open(exe, "r+b") as f:
        f.seek(0x3C)
        pe = int.from_bytes(f.read(4), "little")
        f.seek(pe + 24)
        if f.read(2) != b"\x0b\x02":
            return False
        off = pe + 24 + 70
        f.seek(off)
        dc = int.from_bytes(f.read(2), "little")
        if not (dc & 0x20):
            return False
        f.seek(off)
        f.write((dc & ~0x20).to_bytes(2, "little"))
        return True


def runtime_signature(rtlib):
    """What the exe was linked against. An upgrade replaces the runtime DLL and keeps the exe, so this is the only
    thing that says whether the pair still matches."""
    try:
        st = rtlib.stat()
        return {"size": st.st_size, "mtime": int(st.st_mtime)}
    except OSError:
        return None


def link_stamp_matches(stamp, sig):
    try:
        return sig is not None and json.loads(stamp.read_text(encoding="utf-8")) == sig
    except (OSError, ValueError):
        return False


def install_runtime_signature(install):
    """The runtime an install runs: the package's own copy while it is there, else the one beside the exe."""
    for candidate in (Path(install) / "runtime" / "xc2_rt.dll", Path(install) / "xc2_rt.dll"):
        if candidate.is_file():
            return runtime_signature(candidate)
    return None


def step_link_package(r, repo, install, zig, rtlib, jobs):
    started = r.step("link")
    exe = install / "xc2_boot.exe"
    stamp = install / "build" / "link-stamp.json"
    sig = runtime_signature(rtlib)
    manifest = load_install_manifest(install)
    cpu_var = cpu_variant(zig)
    want_link = link_inputs_hash(repo, install, rtlib, manifest.get("lift"), cpu_var)
    prev_link = manifest.get("link")
    # Keep the exe when the manifest's ABI+stub+lift hash still matches (routine xc2_rt.dll-only updates keep the
    # same 17 C exports and only need the new DLLs copied beside the exe). For legacy installs without a manifest,
    # seed the manifest when the existing exe already matches link-stamp.json or passes PE import/export ABI checks.
    exe_current = False
    if exe.is_file() and want_link is not None:
        if prev_link == want_link:
            exe_current = True
        elif prev_link is None and (link_stamp_matches(stamp, sig) or exe_abi_compatible(exe, repo, install, rtlib)):
            r.log("  manifest: seeded link from existing xc2_boot.exe (legacy install, ABI compatible)")
            manifest["link"] = want_link
            save_install_manifest(install, manifest, dry=r.dry)
            exe_current = True
    elif exe.is_file() and link_stamp_matches(stamp, sig):
        exe_current = True
    if exe_current:
        # Clear high-entropy ASLR on existing installs across upgrades without forcing a full Zig relink.
        if not r.dry and clear_high_entropy_va(exe):
            r.log("  patched %s: cleared HIGH_ENTROPY_VA" % exe.name)
        # The exe is kept across an upgrade; the DLLs still have to land next to it (issue #2).
        copy_runtime_dlls(r, rtlib.parent, install)
        if not r.dry and sig is not None:
            stamp.parent.mkdir(parents=True, exist_ok=True)
            stamp.write_text(json.dumps(sig), encoding="utf-8")
        return r.end_step("link", started, True, "the boot exe is already there")
    lift = install / "build" / "lift"
    obj = install / "build" / "obj-zig"
    inc = repo / "third_party" / "nxrecomp" / "recomp" / "include"
    stub = repo / "xc2" / "src" / "xc2_stub.cpp"
    if not stub.is_file():
        return r.end_step("link", started, False, "the package has no xc2/src/xc2_stub.cpp")
    mcpu = "-mcpu=" + cpu_variant(zig)
    # ZIG_FLAGS starts with -target <triple>; the processor goes right after the triple.
    flags = [str(zig), "c++", "-target", "x86_64-windows-gnu", mcpu] + ZIG_FLAGS[2:]
    srcs = sorted(lift.glob("*/*.cpp"))
    if not srcs:
        return r.end_step("link", started, False, "nothing was lifted")
    objs = []
    todo = []
    for s in srcs:
        o = obj / s.parent.name / (s.stem + ".o")
        o.parent.mkdir(parents=True, exist_ok=True)
        objs.append(o)
        if not o.is_file() or o.stat().st_mtime < s.stat().st_mtime:
            todo.append((s, o))
    if todo:
        n = jobs if jobs > 0 else max(2, (os.cpu_count() or 4) - 1)
        from concurrent.futures import ThreadPoolExecutor, as_completed

        # Precompile <nxrecomp/cxx.h> + "funcs.h" once per multi-shard module so Zig/Clang parses funcs.h once instead of 32x.
        cxx_h = inc / "nxrecomp" / "cxx.h"
        cxx_mtime = cxx_h.stat().st_mtime if cxx_h.is_file() else 0.0
        shard_counts = collections.Counter(s.parent.name for s, _ in todo if s.name.startswith("c_"))
        pch_jobs = []
        pch_map = {}
        for mod_name, count in sorted(shard_counts.items()):
            if count < 2:
                continue
            mod_dir = lift / mod_name
            funcs_h = mod_dir / "funcs.h"
            if not funcs_h.is_file():
                continue
            pch_dir = obj / mod_name
            pch_dir.mkdir(parents=True, exist_ok=True)
            pch_hdr = pch_dir / "nx_pch.h"
            pch_out = pch_dir / "nx_pch.h.pch"
            want_hdr = '#include <nxrecomp/cxx.h>\n#include "funcs.h"\n'
            if not pch_hdr.is_file() or pch_hdr.read_text(encoding="utf-8") != want_hdr:
                pch_hdr.write_text(want_hdr, encoding="utf-8", newline="\n")
            deps_mtime = max(funcs_h.stat().st_mtime, cxx_mtime)
            if pch_out.is_file() and pch_out.stat().st_mtime >= deps_mtime:
                pch_map[mod_name] = pch_out
            else:
                pch_jobs.append((mod_name, mod_dir, pch_hdr, pch_out))
        if pch_jobs:
            def build_pch(item):
                mod_name, mod_dir, pch_hdr, pch_out = item
                p = subprocess.run(flags + ["-I" + str(inc), "-I" + str(mod_dir), "-x", "c++-header", str(pch_hdr), "-o", str(pch_out)],
                                   cwd=str(repo), capture_output=True, text=True, errors="replace")
                return mod_name, pch_out, p.returncode
            with ThreadPoolExecutor(max_workers=min(n, len(pch_jobs))) as ex:
                for mod_name, pch_out, rc in ex.map(build_pch, pch_jobs):
                    if rc == 0 and pch_out.is_file():
                        pch_map[mod_name] = pch_out

        # Longest-processing-time first keeps big main/sdk shards from forming a single-core tail.
        todo.sort(key=lambda item: item[0].stat().st_size, reverse=True)
        r.log("  compiling %d of %d file(s), %d at a time%s"
              % (len(todo), len(srcs), n, " (%d module PCH)" % len(pch_map) if pch_map else ""))

        def build(item):
            s, o = item
            extra = ["-include-pch", str(pch_map[s.parent.name])] if (s.name.startswith("c_") and s.parent.name in pch_map) else []
            p = subprocess.run(flags + extra + ["-I" + str(inc), "-I" + str(s.parent), "-c", str(s), "-o" + str(o)],
                               cwd=str(repo), capture_output=True, text=True, errors="replace")
            return s, p.returncode, (p.stderr or "")[-400:]

        done = 0
        with ThreadPoolExecutor(max_workers=n) as ex:
            futures = [ex.submit(build, t) for t in todo]
            for f in as_completed(futures):
                s, rc, err = f.result()
                done += 1
                r.note_progress("compile %d/%d" % (done, len(todo)), 100.0 * done / len(todo))
                if rc != 0:
                    r.log("  %s failed: %s" % (s.name, err.strip()[:300]))
                    return r.end_step("link", started, False, "compiling %s failed" % s.name)
        r.log("  compiled %d file(s)" % len(todo))
    rsp = obj / "objects.rsp"
    if not r.dry:
        obj.mkdir(parents=True, exist_ok=True)
        rsp.write_text("".join('"%s"\n' % o for o in objs), encoding="ascii")
    # Skip generating xc2_boot.pdb (469 MB) since sweep_leftovers deletes it right after link.
    argv = ([str(zig), "c++", "-target", "x86_64-windows-gnu", "-O1", "-w", "-fno-exceptions", "-fno-rtti",
             "-Wl,--strip-debug", "-DXC2_MODULES_7", "@" + str(rsp), str(stub),
             "-I" + str(inc), "-I" + str(stub.parent), str(rtlib), "-o", str(exe)])
    if r.run(argv) != 0:
        return r.end_step("link", started, False, "the exe did not link")
    if not r.dry and exe.is_file() and clear_high_entropy_va(exe):
        r.log("  patched %s: cleared HIGH_ENTROPY_VA" % exe.name)
    copy_runtime_dlls(r, rtlib.parent, install)
    finalize_m3_state(r, install)
    if want_link is not None:
        manifest["link"] = want_link
        save_install_manifest(install, manifest, dry=r.dry)
    if not r.dry and sig is not None:
        stamp.parent.mkdir(parents=True, exist_ok=True)
        stamp.write_text(json.dumps(sig), encoding="utf-8")
    return r.end_step("link", started, True)

def junction(link, target, r):
    if link.exists() or link.is_symlink():
        return 0
    r.log("  linking %s -> %s" % (link, target))
    if r.dry:
        return 0
    link.parent.mkdir(parents=True, exist_ok=True)
    return subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(target)],
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode


def link_boot_romfs(r, repo, install):
    """Give play-301.cmd the relative romfs path it already looks for.

    That script runs from <repo>/xc2/build/boot and sets NXRT_FS_ROMFS=game/romfs-301 - a path relative to that
    working directory, which is what keeps the tree portable. Nothing created it, so an install finished "ready to
    play" and the first launch died: "fs: RomFS folder 'game/romfs-301' cannot be served" -> svcBreak, exit 4.
    The relative path is the right shape, so the install supplies it rather than the launcher learning a new one.
    """
    romfs = install / "romfs"
    if not romfs.is_dir():
        r.log("  no %s; play-301.cmd will not find a romfs" % romfs)
        return 0
    return junction(repo / "xc2" / "build" / "boot" / "game" / "romfs-301", romfs, r)


def romfs_missing(looked_near, romfs_in):
    """Say which folder was checked and which file is missing, instead of "pass --romfs".

    The check is (folder / "bf2.ard"), not "does the folder exist", so a romfs that sits right next to exefs still
    fails when the archive is not inside it - and the old wording ("no romfs near <exefs>; pass --romfs") pointed at
    the wrong thing twice: it ignored that --romfs had already been passed, and it blamed the folder rather than the
    missing bf2.ard (issue #11, 2026-11-19).
    """
    if romfs_in:
        return ("the romfs at %s has no bf2.ard; the archive is required, and it has to sit directly in that folder"
                % romfs_in)
    return ("no bf2.ard in %s, %s, %s or %s; pass --romfs <the folder that holds bf2.ard>"
            % (looked_near, looked_near / "romfs", looked_near.parent, looked_near.parent / "romfs"))


NS2_UPGRADE_PACK_ID = "0100e95004039062"


def _is_title_id(name):
    return len(name) == 16 and all(c in "0123456789abcdefABCDEF" for c in name)


def _update_version_in_path(path):
    parts = [p.lower() for p in path.parts]
    best = 0
    for i in range(len(parts) - 1):
        if parts[i] == "updates" and parts[i + 1].isdigit():
            best = max(best, int(parts[i + 1]))
    return best


def find_addon_romfs(addon_dir, max_depth=6):
    """Locate the best `romfs/` inside a title-ID folder, supporting both flat `<id>/romfs` and
    raw nested layouts such as `<id>/0/updates/65536/program/0/romfs` or `<id>/0/data/0/romfs`."""
    direct = addon_dir / "romfs"
    if direct.is_dir():
        return direct
    found = []
    stack = [(addon_dir, 0)]
    while stack:
        cur, depth = stack.pop()
        if depth >= max_depth:
            continue
        try:
            entries = list(cur.iterdir())
        except OSError:
            continue
        for entry in entries:
            if not entry.is_dir():
                continue
            low = entry.name.lower()
            if low == "romfs":
                found.append((entry, depth + 1))
            elif low != "exefs":
                stack.append((entry, depth + 1))
    if not found:
        return None

    def rank(item):
        p, depth = item
        ard = p / "bf2.ard"
        has_ard = ard.is_file()
        ard_size = ard.stat().st_size if has_ard else 0
        return (1 if has_ard else 0, ard_size, _update_version_in_path(p), -depth)

    return max(found, key=rank)[0]


def find_ns2_upgrade_romfs(root):
    """In an un-trimmed raw Switch 2 dump, the full ~24 GB NS2 3.0.1 RomFS lives inside
    `0100e95004039062` (e.g. `0100e95004039062/0/updates/65536/program/0/romfs`)."""
    roots = [root]
    if root.parent != root:
        roots.append(root.parent)
    for base in roots:
        for parent in (base / "dlc", base):
            if not parent.is_dir():
                continue
            try:
                entries = list(parent.iterdir())
            except OSError:
                continue
            for entry in entries:
                if entry.is_dir() and entry.name.lower() == NS2_UPGRADE_PACK_ID:
                    romfs = find_addon_romfs(entry)
                    if romfs is not None and (romfs / "bf2.ard").is_file():
                        return romfs
    return None


def find_romfs(dump, explicit):
    # --romfs wins when it is given. Measured 2026-11-17: a run that passed --romfs looked like it had linked the
    # maintainer tree instead, and had not - Path.resolve() dereferences a junction, and the folder passed in WAS a
    # junction to that one, so the install's link named the real target. Both names reach the same files.
    if explicit:
        return explicit if (explicit / "bf2.ard").is_file() else None
    upgrade = find_ns2_upgrade_romfs(dump)
    if upgrade is not None:
        return upgrade
    for candidate in (dump, dump / "romfs", dump.parent, dump.parent / "romfs"):
        if (candidate / "bf2.ard").is_file():
            return candidate
    return None


def discover_dlc_romfs(src_dir):
    """Return sorted list of (lowercase_title_id, romfs_path) for valid DLCs in `src_dir`,
    skipping the NS2 upgrade pack (0100e95004039062)."""
    if not src_dir or not src_dir.is_dir():
        return []
    found = {}
    try:
        entries = list(src_dir.iterdir())
    except OSError:
        return []
    for entry in entries:
        if not entry.is_dir():
            continue
        name = entry.name
        if not _is_title_id(name) or name.lower() == NS2_UPGRADE_PACK_ID:
            continue
        romfs = find_addon_romfs(entry)
        if romfs is not None:
            found[name.lower()] = romfs
    return sorted(found.items())


def _is_reparse_point(path):
    if path.is_symlink():
        return True
    is_junc = getattr(path, "is_junction", None)
    if callable(is_junc):
        try:
            return is_junc()
        except OSError:
            return False
    return False


def link_dlc(install, r, dump_hint=None):
    """Link each valid add-on's `romfs/` into `<install>/game/dlc-301/<id>/romfs` via per-DLC junctions.

    This keeps the player's dump read-only with zero file copies while normalizing raw nested DLC dumps
    (`<id>/0/.../romfs` or DLC folders placed directly in the dump root) into the flat `<id>/romfs`
    layout that the runtime (`misc.cpp`) expects, and always skips `0100e95004039062`.
    """
    candidates = []
    env_src = os.environ.get("XC2_AOC_DIR", "")
    if env_src:
        candidates.append(Path(env_src))
    if dump_hint is not None:
        hint = Path(dump_hint)
        candidates.extend([hint / "dlc", hint])
        if hint.parent != hint:
            candidates.extend([hint.parent / "dlc", hint.parent])
    src = None
    addons = []
    for cand in candidates:
        found = discover_dlc_romfs(cand)
        if found:
            src = cand
            addons = found
            break
    if not addons:
        r.log("  dlc: none in the dump (the game runs without add-on content)")
        return
    dlc_root = install / "game" / "dlc-301"
    if not r.dry:
        if _is_reparse_point(dlc_root):
            # Replace legacy whole-folder junction with a real directory holding per-DLC junctions.
            os.rmdir(dlc_root)
        dlc_root.mkdir(parents=True, exist_ok=True)
    linked = 0
    for tid, target_romfs in addons:
        if junction(dlc_root / tid / "romfs", target_romfs, r) == 0:
            linked += 1
    r.log("  dlc: linked %d add-on(s) from %s -> %s" % (linked, src, dlc_root))


def step_ingest(r, repo, install, nup, dump, exefs_in, romfs_in):
    started = r.step("ingest")
    exefs = install / "exefs"
    romfs = install / "romfs"
    if have_modules(exefs) and romfs.is_dir():
        link_dlc(install, r, dump_hint=dump or exefs_in)
        return r.end_step("ingest", started, True, "the game files are already installed")
    py = os.environ.get("XC2_PYTHON") or sys.executable
    extract = repo / "tools" / "analysis" / "nup_extract.py"
    if nup:
        if r.run([py, extract, "--nup", nup, "--pkg", "0", "--part", "exefs", "--out", exefs]) != 0:
            return r.end_step("ingest", started, False, "the exefs could not be unpacked (is zstandard installed?)")
        if not (romfs / "bf2.ard").is_file():
            if r.run([py, extract, "--nup", nup, "--pkg", "0", "--part", "romfs", "--out", romfs]) != 0:
                return r.end_step("ingest", started, False, "the romfs could not be unpacked")
    elif exefs_in:
        if not have_modules(exefs_in):
            return r.end_step("ingest", started, False, "no NSO modules in %s" % exefs_in)
        rom = find_romfs(exefs_in, romfs_in)
        if rom is None:
            return r.end_step("ingest", started, False, romfs_missing(exefs_in, romfs_in))
        if junction(exefs, exefs_in, r) != 0:
            return r.end_step("ingest", started, False, "could not link exefs")
        if junction(romfs, rom, r) != 0:
            return r.end_step("ingest", started, False, "could not link romfs")
        link_dlc(install, r, dump_hint=exefs_in)
    elif dump:
        found = dump if (dump / "main").is_file() else dump / "exefs"
        if not have_modules(found):
            return r.end_step("ingest", started, False, "no NSO modules in %s" % found)
        if junction(exefs, found, r) != 0:
            return r.end_step("ingest", started, False, "could not link exefs")
        rom = find_romfs(dump, romfs_in)
        if rom is None:
            return r.end_step("ingest", started, False, romfs_missing(dump, romfs_in))
        if junction(romfs, rom, r) != 0:
            return r.end_step("ingest", started, False, "could not link romfs")
        link_dlc(install, r, dump_hint=dump)
    else:
        return r.end_step("ingest", started, False, "give --nup or --dump")
    return r.end_step("ingest", started, True)


# The switches the retail 3.0.1 build needs before it runs at all. xc2/src/xc2_boot.cpp reads the first two - they arm
# hooks at fixed sdk offsets, which is why they stay opt-in rather than default-on - and the sdk reads the rest.
# tools/run/play-301.cmd sets the same list for a checkout. A headless run without them dies in 0.6 s, names no block,
# and looks exactly like "the archive scan was already complete": measured 2026-11-18, 0 of 5 collected.
GAME_ENV = {
    "XC2_301_BOOTPARAM": "1",
    "XC2_301_VAMM": "1",
    "XC2_301_OPUS": "1",
    "XC2_301_AUDIO": "1",
    "NXRT_PROCESS_SYSTEM_RESOURCE_SIZE": "0x6000000",
    "NXRT_TITLE_DISPLAY_VERSION": "3.0.1",
    "NXRT_IPC_STUB_RESULT": "0",
    "NXRT_MEMORY_LIMIT_GB": "24",
    "NXRT_XC2_FPS": "60",
    "NXRT_NVN_PRESENT_INTERVAL_HZ": "60",
    "NXRT_NVN_QUAD_SPLIT": "16",
    # The light list's registration mask is derived after every list draw ([nvn] earlyz_mask, on by default) but only
    # for the shaders in [nvn] earlyz_fs, whose default is the eleven of the old list. The game's list now has more,
    # so those drew without a rebuilt mask and the black rectangle came back (measured 2026-11-18: "fs d3aa3541 writes
    # a storage image but is not in earlyz_fs, so the mask is not rebuilt"). auto = every shader, gated by the format
    # test inside the rebuild.
    "NXRT_NVN_EARLYZ_FS": "auto",
    # NXRT_NVN_PRESENT_AHEAD is deliberately not here: an env value wins over everything, and whether the guest may run a
    # frame ahead depends on the GPU ([nvn] present_ahead = auto, xc2_boot.cpp: only while [nvn] mirror_capture is on).
    "NXRT_NVN_EARLYZ_MASK": "1",
    # The deferred lighting pass is one full-screen triangle with cull BACK; with the wrong front face it is culled and
    # the whole frame stays at the G-buffer's albedo (flat, untextured, black-looking). tools/run/play-301.cmd uses 2;
    # the runtime default is 1, so a launcher that forgets this renders the scene wrong.
    "NXRT_NVN_FRONT_FACE_MODE": "2",
    # Only the languages the game has: American English / Latin American Spanish fall back to British / Spanish.
    "NXRT_TITLE_SUPPORTED_LANGUAGES": "Japanese,BritishEnglish,French,German,Italian,Spanish,"
                                       "Chinese,Taiwanese,Korean,SimplifiedChinese,TraditionalChinese",
}


def launch_env(install):
    """Write <install>/launch.env from GAME_ENV and hand it back.

    The launcher starts the game on its own (the package must not need anything outside itself), so the switches have
    to exist somewhere both it and the headless prefill can read. Keeping them in a file the engine writes avoids the
    second copy in Rust that would drift the first time one side changed.
    """
    path = install / "launch.env"
    text = ("# The switches the retail 3.0.1 build needs. Written by tools/setup/xc2_setup.py from GAME_ENV on every\n"
            "# setup and read by the launcher (Play) and by the headless prefill, so there is one list, not two.\n")
    for k in sorted(GAME_ENV):
        text += "%s=%s\n" % (k, GAME_ENV[k])
    path.write_text(text, encoding="utf-8", newline="\n")
    return dict(GAME_ENV)


def step_shaders_package(r, repo, install, py, jobs):
    """Translate the game's shaders into the cache the runtime reads.

    The shaders are embedded in the romfs's bf2.ard, not in nn::gfx containers, so this walks the archive's embedded
    group layout (tools/shader-cache: ard_index, extract_embedded, translate, install). The cache lands in
    <install>/shader_cache/spirv, which is where [nvn] shader_cache points. A package ships no cache: it is derived
    from the player's own copy of the game.

    The shader cache is append-only (12,137 complete programs = 12,132 from bf2.ard + 5 runtime-generated prefill
    blocks). When the shader translator changes, existing archive shaders are refreshed in-place (--refresh) without
    deleting the cache, and any stale spirv.preload.pack is explicitly dropped so the runtime rebuilds it.
    """
    started = r.step("shaders")
    spirv = install / "shader_cache" / "spirv"
    harness = repo / "tools" / "spirv_harness" / "spirv_harness.exe"
    # Two halves, two markers: results.tsv says the archive was walked, "prefilled" says the headless run named what no
    # file contains. They are separate on purpose - a setup that dies between them used to look finished (12,132 in the
    # cache, 5 missing, black title) and every later setup skipped the step entirely.
    scanned = (spirv / "results.tsv").is_file() and any(spirv.glob("*.spv"))
    prefilled = install / "build" / "prefilled"
    sig = install_runtime_signature(install)
    prefill_stamp = install / "build" / "prefill-stamp.json"

    manifest = load_install_manifest(install)
    want_shaders = shader_inputs_hash(repo, install)
    want_prefill = prefill_inputs_hash(want_shaders)
    prev_shaders = manifest.get("shaders")
    prev_prefill = manifest.get("prefill")

    # Legacy seeding: an existing install without an install-manifest.json must not be forced to re-translate all
    # 12,132 archive shaders on its first manifest-enabled upgrade.
    if prev_shaders is None and scanned:
        r.log("  manifest: seeded shaders from existing cache (legacy install)")
        manifest["shaders"] = want_shaders
        prev_shaders = want_shaders
        if prefilled.is_file() and (not prefill_stamp.is_file() or link_stamp_matches(prefill_stamp, sig)):
            manifest["prefill"] = want_prefill
            prev_prefill = want_prefill
        save_install_manifest(install, manifest, dry=r.dry)

    need_refresh = scanned and (prev_shaders != want_shaders)
    if prev_prefill is not None:
        prefill_stale = prefilled.is_file() and (prev_prefill != want_prefill or need_refresh)
    else:
        # Legacy install whose prefill-stamp.json was already mismatched against the runtime (#83).
        prefill_stale = prefilled.is_file() and prefill_stamp.is_file() and not link_stamp_matches(prefill_stamp, sig)
    if prefill_stale:
        r.log("  shader prefill inputs changed; running the headless prefill pass again (the cache is kept)")
        if not r.dry:
            prefilled.unlink(missing_ok=True)
    if scanned and not need_refresh and prefilled.is_file() and not prefill_stale:
        if not r.dry and sig is not None and not link_stamp_matches(prefill_stamp, sig):
            prefill_stamp.parent.mkdir(parents=True, exist_ok=True)
            prefill_stamp.write_text(json.dumps(sig), encoding="utf-8")
        return r.end_step("shaders", started, True, "the shader cache is already there")
    if not scanned or need_refresh:
        tool = first_file(repo / "tools" / "shader" / "cache" / "build_cache.py", repo / "tools" / "shader-cache" / "build_cache.py")
        ard = install / "romfs" / "bf2.ard"
        if not tool.is_file():
            return r.end_step("shaders", started, False, "the package has no tools/shader-cache/build_cache.py")
        if not harness.is_file():
            return r.end_step("shaders", started, False, "the package has no shader harness")
        if not ard.is_file():
            return r.end_step("shaders", started, False, "no bf2.ard under " + str(install / "romfs"))
        n = jobs if jobs > 0 else max(2, (os.cpu_count() or 4) - 1)
        if need_refresh:
            r.log("  shader translator or archive changed; re-translating archive shaders in-place (append-only; takes about as long as first-time setup)")
        else:
            r.log("  reading the archive and translating what the cache lacks; this is the long step")
        # The progress filter drops everything that is not a stage line, which also dropped the child's traceback:
        # a failing shader step used to say "the shader chain failed" and nothing else (issue #5, 2026-11-18). Keep the
        # tail and print it when the child exits non-zero.
        tail = collections.deque(maxlen=40)
        progress = shader_progress(r)

        def keep(line, tail=tail, progress=progress):
            tail.append(line)
            progress(line)

        cmd = [str(py), str(tool), "--ard", str(ard), "--cache", str(spirv), "--harness", str(harness),
               "--work", str(install / "build" / "shader-cache"), "--jobs", str(n), "--no-val"]
        if need_refresh:
            cmd.append("--refresh")
        rc = r.run(cmd, cwd=repo, on_line=keep)
        if rc != 0:
            r.log("  the shader chain exited with code %d; its last lines:" % rc)
            for ln in tail:
                r.log("  | " + ln.rstrip()[:300])
            return r.end_step("shaders", started, False, "the shader chain failed (exit %d); see the lines above" % rc)
        r.log("  %d shader(s) from the archive" % len(list(spirv.glob("*.spv"))))
        if need_refresh:
            # spirv.preload.pack is read before loose .spv files; explicitly drop it after refreshing shaders.
            drop_preload_pack(r, spirv)
        manifest["shaders"] = want_shaders
        save_install_manifest(install, manifest, dry=r.dry)
    else:
        # Re-walking the archive only retries the 22 blocks the translator rejects, and build_cache.py treats "not one
        # shader came out" as a hard failure - measured 2026-11-18. The scan half is done; only the prefill is missing.
        r.log("  the archive scan is already there; only the headless prefill is missing")
    programs = spirv.parent / "programs.bin"
    tsv = install / "build" / "shader-cache" / "files.tsv"
    ard = install / "romfs" / "bf2.ard"
    if not programs.is_file() and ard.is_file() and tsv.is_file():
        ext_tool = first_file(repo / "tools" / "shader" / "cache" / "extract_embedded.py",
                              repo / "tools" / "shader-cache" / "extract_embedded.py")
        if ext_tool.is_file():
            n = jobs if jobs > 0 else max(2, (os.cpu_count() or 4) - 1)
            r.log("  extracting paired shader programs list for pipeline prewarming...")
            r.run([str(py), str(ext_tool), "--ard", str(ard), "--tsv", str(tsv),
                   "--programs", str(programs), "--programs-only", "--jobs", str(n)], cwd=repo)
    note = prefill_package(r, repo, install, py, harness, spirv, jobs=jobs)
    r.log("  %d shader(s) in the cache" % len(list(spirv.glob("*.spv"))))
    if not prefilled.is_file():
        note = note or "the headless prefill named no block; the cache has the archive's share only"
        mods = code_patch_mods(install)
        if mods:
            note += ("; code patches from %s are compiled into this build - disable the mod and run Set up again "
                     "if the patch was written for another build" % ", ".join(mods))
        return r.end_step("shaders", started, False, note)
    manifest["shaders"] = want_shaders
    manifest["prefill"] = want_prefill
    save_install_manifest(install, manifest, dry=r.dry)
    if not r.dry and sig is not None:
        prefill_stamp.parent.mkdir(parents=True, exist_ok=True)
        prefill_stamp.write_text(json.dumps(sig), encoding="utf-8")
    exe = install / "xc2_boot.exe"
    if exe.is_file() and programs.is_file():
        patched = install / "build" / "exefs-patched"
        exefs_dir = patched if (patched / "m3_state.json").is_file() and (patched / "main").is_file() else install / "exefs"
        env = launch_env(install)
        env["NXRT_NVN_SHADER_CACHE"] = str(spirv)
        r.log("  precompiling GPU pipeline cache from %s..." % programs.name)
        r.note_progress("precompiling GPU pipeline cache")
        rc = r.run([str(exe), "--exefs", str(exefs_dir), "--link", "host", "--gfx", "headless",
                    "--prepare-shaders", "--log", str(install / "setup" / "prepare.log")],
                   cwd=install, env=env)
        if rc != 0:
            r.log("  pipeline precompile returned exit %d (will retry automatically on launch)" % rc)
    return r.end_step("shaders", started, True)


# The runtime's words for a start-up that died, and the one thing the player can do about each. A GPU driver older than
# Vulkan 1.3 is the common one: the game needs 1.3, the driver refuses the instance, and before this table existed the
# setup page said "the headless run exited with code 1" and stopped there.
VULKAN_HINTS = (
    # markers in the runtime's log, a short form for the step's note, and the line the player reads
    (("VK_ERROR_INCOMPATIBLE_DRIVER", "vkCreateInstance failed (-9)", "Vulkan 1.3 is required"),
     "the GPU driver does not offer Vulkan 1.3",
     "the GPU driver does not offer Vulkan 1.3 and the game needs it: install the newest driver for the card and run the "
     "setup again (the Vulkan runtime comes with the driver)"),
    (("no Vulkan loader found",),
     "vulkan-1.dll is missing",
     "vulkan-1.dll is not on this machine: install the GPU driver - it ships the Vulkan loader"),
    (("no usable Vulkan 1.3 GPU",),
     "no GPU here offers Vulkan 1.3",
     "Vulkan starts, but no GPU here offers 1.3 with the features the renderer needs: install the newest driver for the card"),
    (("no Vulkan physical device",),
     "the driver lists no GPU",
     "the driver loaded but lists no GPU - a remote desktop session or a VM without GPU passthrough reports that"),
)


def log_tail(path, lines=400):
    """The last few hundred lines of a log the game wrote. It is written continuously, so it can be megabytes."""
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            f.seek(max(0, f.tell() - 128 * 1024))
            text = f.read().decode("utf-8", "replace")
    except OSError:
        return []
    return text.splitlines()[-lines:]


def log_message(line):
    """A runtime log line without its "[  1.234 I t5678 -]" prefix."""
    return line.split("] ", 1)[1].strip() if "] " in line else line.strip()


def log_seconds(tail):
    """The timestamp of the last line, or None: a run that stops at 1.2 s never reached the renderer."""
    for line in reversed(tail):
        m = re.match(r"\[\s*([\d.]+)\s", line)
        if m:
            try:
                return float(m.group(1))
            except ValueError:
                return None
    return None


def exit_text(code):
    """An exit code as the page should print it: a Windows crash code is hex, and reads as noise in base ten."""
    if code >= 0x80000000 or code < 0:
        return "it crashed (0x%08X)" % (code & 0xFFFFFFFF)
    return "it exited with code %d" % code


def prefill_failure(log_path):
    """(the line that explains the stop, its short form, what to do about it) for a prefill that collected nothing.

    Three things end a prefill with no block collected and only one of them is a finished prefill: start-up died (a GPU
    driver older than Vulkan 1.3 is the usual cause), the cap came before the renderer drew anything, or there really
    was nothing missing. The runtime writes its own log - the prefill's own output goes to NUL - so the reason has been
    in this file all along; a failed run reached the setup page as a bare exit code, which is what "stuck at the prefill
    step" looked like in the reports. Returns ("", "", "") when the log is silent, so the caller keeps the old answer.
    """
    tail = log_tail(log_path)
    if not tail:
        return "", "", ""
    text = "\n".join(tail)
    reason = ""
    for line in tail:
        if "device creation failed:" in line or "graphics device:" in line or "runtime start-up failed" in line:
            reason = log_message(line)
    if not reason:
        for line in reversed(tail):
            # "frame times: no frame was presented yet" is the gauge's summary of a run that never presented, not the reason
            # it never did - it was picked as the cause of a prefill failure and told the reporter the wrong thing (issue #25,
            # 2026-11-21). Skip report lines; the reason is an error above them.
            if "frame times:" in line or "no frame presented in the last" in line:
                continue
            if re.match(r"^\[\s*[\d.]+\s+[EF]\b", line):
                reason = log_message(line)
                break
    if not reason:
        stopped = log_seconds(tail)
        if stopped is not None and stopped < 5.0:
            reason = "the run stopped %.1f s in, before the renderer drew anything" % stopped
    short, hint = next(((s, h) for markers, s, h in VULKAN_HINTS if any(m in text for m in markers)), ("", ""))
    return reason, short, hint


def prefill_package(r, repo, install, py, harness, spirv, seconds=40, jobs=0):
    """Run the game headless once and translate whatever the archive scan could not name.

    A handful of blocks in 3.0.1 are in no file at all - the game builds them at run time - so scanning the archive
    can never be complete on its own (measured: 12,154 from the archive, 5 more that exist nowhere). The runtime
    already knows which blocks it is missing, so one short headless run collects exactly those.
    """
    exe = install / "xc2_boot.exe"
    if not exe.is_file():
        r.log("  prefill: no xc2_boot.exe yet, skipping")
        return ""
    missing = install / "build" / "missing"
    prefilled = install / "build" / "prefilled"
    if missing.is_dir():
        shutil.rmtree(missing, ignore_errors=True)
    missing.mkdir(parents=True, exist_ok=True)
    env = {
        "NXRT_NVN_DUMP_MISSING": str(missing),
        # Names the samplers that read the stencil plane through a float one, one line per (shader, binding); the patcher
        # below turns those into legal uint samplers. A fresh cache has no such lines without this (handover 125).
        "NXRT_NVN_DS_DEBUG": "1",
        "NXRT_NVN_SHADER_CACHE": str(spirv),
        # <install>/romfs, not game/romfs-301: that junction is made by the done step, which runs *after* this one, so a
        # fresh install had nothing there and the game died on RomFS 0x7D402 five seconds in (measured 2026-11-18).
        "NXRT_FS_ROMFS": str(install / "romfs"),
        "NXRT_FS_SAVE_DIR": str(install / "save"),
    }
    log = install / "setup" / "prefill.log"
    patched = install / "build" / "exefs-patched"
    exefs_dir = patched if (patched / "m3_state.json").is_file() and (patched / "main").is_file() else install / "exefs"
    argv = [str(exe), "--exefs", str(exefs_dir), "--link", "host", "--gfx", "headless", "--log", str(log)]
    r.log("  prefill: the game runs headless for up to %d s to name the blocks no file contains" % seconds)
    full = os.environ.copy()
    full.update(launch_env(install))
    full.update(env)
    try:
        p = subprocess.Popen(argv, cwd=str(install), env=full,
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except OSError as e:
        r.log("  prefill: could not start the game (%s); the cache stays as the archive left it" % e)
        return "the headless prefill could not start: %s" % e
    started_wait = time.time()
    deadline = started_wait + seconds
    seen = 0
    ds_count = 0
    log_pos = 0
    last_activity = started_wait
    while time.time() < deadline and p.poll() is None:
        n = len(list(missing.glob("code_*.bin")))
        if n != seen:
            seen = n
            last_activity = time.time()
            r.note_progress("prefill: %d block(s) collected" % n)
        if log.is_file():
            try:
                with open(log, "rb") as lf:
                    lf.seek(log_pos)
                    chunk = lf.read()
                    log_pos += len(chunk)
                hits = chunk.count(b"ds_debug:")
                if hits:
                    ds_count += hits
                    last_activity = time.time()
            except OSError:
                pass
        now_t = time.time()
        if (seen > 0 or ds_count > 0) and (now_t - started_wait >= 12.0) and (now_t - last_activity >= 6.0):
            r.log("  prefill: settled after %.1f s (%d block(s), %d ds line(s))" % (now_t - started_wait, seen, ds_count))
            break
        time.sleep(0.5)
    finished = p.poll()
    killed = finished is None
    if killed:
        p.kill()
        p.wait()
    blocks = sorted(missing.glob("code_*.bin"))
    r.log("  prefill: collected %d block(s)" % len(blocks))
    if not blocks:
        # Nothing missing is the normal second run, and a failed one must not look like it: that is how the missing
        # switches went unnoticed. The failure is the case the reports called "stuck at the prefill step" - the reason
        # is in the run's own log, and the page used to show only an exit code.
        reason, short, hint = prefill_failure(log)
        if reason or hint or (not killed and finished != 0):
            r.log("  prefill: the headless run named no block - the cache stays as the archive left it")
            if finished is not None and finished != 0:
                r.log("  prefill: " + exit_text(finished))
            if reason:
                r.log("  prefill: " + reason)
            if hint:
                r.log("  prefill: " + hint)
            r.log("  prefill: the run's own log: %s" % log)
            how = "exited with code %d" % finished if (finished is not None and finished != 0) else "was stopped"
            # The cap is how long the pass may run, not how long it did: a run that died 2 s in was reported as
            # "after 40 s", which reads like a timeout and hides that it crashed on start-up (issue #75).
            ran = max(0.0, time.time() - started_wait)
            return "the headless prefill %s after %.0f s without naming a block: %s" % (how, ran, short or reason[:96] or hint[:96])
        retype_ds_samplers(r, repo, py, spirv, log)
        prefilled.write_text("0 collected: nothing was missing\n", encoding="utf-8")
        return ""
    n = jobs if jobs > 0 else max(2, (os.cpu_count() or 4) - 1)
    rc = r.run([str(py), str(first_file(repo / "tools" / "shader" / "cache" / "build_cache.py", repo / "tools" / "shader-cache" / "build_cache.py")), "--blocks", str(missing),
                "--cache", str(spirv), "--harness", str(harness), "--jobs", str(n), "--no-val"], cwd=repo,
               on_line=shader_progress(r))
    if rc != 0:
        r.log("  prefill: the collected blocks did not translate; the game will still run")
        return ""
    retype_ds_samplers(r, repo, py, spirv, log)
    prefilled.write_text("%d collected and translated\n" % len(blocks), encoding="utf-8")
    return ""

def retype_ds_samplers(r, repo, py, spirv, log):
    """Retype the stencil-plane samplers the prefill's ds_debug lines named (format-07753).

    A cache translated from the archive alone declares those samplers as float while the runtime binds an S8_UINT
    view, so the descriptor never matches and the deferred lighting's class mask reads black: characters and the
    background come out untextured. The old flow needed a second full play session to get these lines; the prefill
    run produces them, so the first launch is already correct.
    """
    patch = repo / "tools" / "shader" / "patch_ds_samplers.py"
    if not patch.is_file():
        r.log("  prefill: no tools/shader/patch_ds_samplers.py in the package; stencil samplers stay as they are")
        return
    rc = r.run([str(py), str(patch), "--cache", str(spirv), "--log", str(log)], cwd=repo)
    if rc == 0:
        r.log("  prefill: stencil-plane samplers retyped from the ds_debug lines")
    else:
        r.log("  prefill: no stencil sampler needed retyping (exit %d)" % rc)


def shader_progress(r):
    """build_cache.py reports stages and streamed counters; pass those on so the page shows movement."""
    stage = ""

    def on_line(line):
        nonlocal stage
        s = line.strip()
        if s[:1] == "[" and any(k in s for k in ("index", "extract", "plan", "stage", "translate", "install", "pack")):
            stage = s.split("]", 1)[0].lstrip("[")
            r.note_progress(s[:140])
            return
        m = re.match(r"^(\d+)/(\d+)\b", s)
        if m:
            done, total = int(m.group(1)), int(m.group(2))
            pct = (100.0 * done / total) if total > 0 else None
            label = stage or ("extract" if "groups=" in s else "translate")
            r.note_progress("%s %d/%d" % (label, done, total), pct)
    return on_line

def step_done_package(r, repo, install):
    """The package runs the exe from <install>, which looks for game/romfs-301 there (not under a build folder)."""
    started = r.step("done")
    # Also here, not only in the shaders step: a package whose cache was already complete skips that step entirely,
    # and the launcher still needs the switches.
    launch_env(install)
    if not r.dry:
        gone = sweep_leftovers(install)
        if gone:
            r.log("  removed after the build: " + ", ".join(sorted(gone)))
        made = write_ini(install, repo)
        if made:
            r.log("  wrote nxrt.ini")
        unshadowed = unshadow_ini(install)
        if unshadowed:
            r.log("  nxrt.ini: dropped " + ", ".join(unshadowed) + " (the F1 page owns those; see #57 / #73)")
    game = install / "game"
    if not r.dry:
        game.mkdir(parents=True, exist_ok=True)
    junction(game / "romfs-301", install / "romfs", r)
    if (install / "nup_dump" / "dlc").is_dir():
        link_dlc(install, r, dump_hint=install / "nup_dump")
    if not r.dry:
        (install / "setup_state.tsv").write_text("mprs_setup_state\t1\nmeta\tfinished\t1\n", encoding="utf-8")
    r.write_status(state="done", note="ready to play")
    return r.end_step("done", started, True, "ready to play")


# The template used to seed keys that the F1 page owns, and nxrt.ini wins over controls.ini (play.cpp merge_env), so
# the player's choice was replaced on every launch: [hdr] mode = auto pinned HDR (#57), [gfx] window_width/height pinned
# the window size (#73). New installs no longer get them. An nxrt.ini written by an older template still holds them, so
# setup drops each one while it still carries exactly the old default - a value the player set is left alone.
SHADOWED_DEFAULTS = (("hdr", "mode", "auto"), ("gfx", "window_width", "1920"), ("gfx", "window_height", "1080"))


def _shadowed_key(section, line):
    body = line.split(";")[0].strip()
    if "=" not in body:
        return None
    name, _, value = body.partition("=")
    name = name.strip().lower()
    value = value.strip().strip('"').lower()
    for sec, key, default in SHADOWED_DEFAULTS:
        if section == sec and name == key and value == default:
            return sec + "/" + key
    return None


def unshadow_ini(install):
    """Drop the keys an older template seeded with their default, so the F1 page decides those settings again."""
    ini = install / "nxrt.ini"
    if not ini.is_file():
        return []
    lines = ini.read_text(encoding="utf-8", errors="replace").splitlines(keepends=True)
    section = ""
    kept = []
    dropped = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            section = stripped[1:-1].strip().lower()
            kept.append(line)
            continue
        hit = _shadowed_key(section, line)
        if hit:
            dropped.append(hit)
        else:
            kept.append(line)
    if not dropped:
        return []
    ini.write_text("".join(kept), encoding="utf-8", newline="\n")
    return dropped


def write_ini(install, repo):
    """Give a fresh install its nxrt.ini before anything runs: the launcher edits this file, it does not invent it.

    The template lives in the checkout (config/nxrt.ini.in) and in a package (engine/nxrt.ini.in), and every key
    in it is checked against config/keys.tsv by tools/ci/config_table.py --check.
    """
    ini = install / "nxrt.ini"
    if ini.is_file():
        return None
    template = None
    for cand in (repo / "engine" / "nxrt.ini.in", repo / "config" / "nxrt.ini.in"):
        if cand.is_file():
            template = cand
            break
    if template is None:
        return None
    dlc = install / "dlc"
    for cand in (install / "game" / "dlc-301", install / "nup_dump" / "dlc", install / "dlc"):
        if cand.is_dir():
            dlc = cand
            break
    values = {
        "exefs": install / "exefs",
        "romfs": install / "romfs",
        "mods": install / "mods",
        "dlc": dlc,
        "shader_cache": install / "shader_cache" / "spirv",
        "bc7_cache": install / "shader_cache" / "bc7",
    }
    text = template.read_text(encoding="utf-8")
    for key, value in values.items():
        text = text.replace("{%s}" % key, str(value).replace("\\", "/"))
    ini.write_text(text, encoding="utf-8", newline="\n")
    return ini


def sweep_leftovers(install):
    """A link leaves a .pdb next to the exe and the cache builder keeps a results.tsv backup; neither ships.

    Measured on a fresh 0.1.3 install: xc2_boot.pdb was 469 MB of a 2.5 GB tree, and the backup 2.5 MB.
    """
    gone = []
    for path in list(install.glob("*.pdb")) + list((install / "shader_cache" / "spirv").glob("results.tsv.bak-*")):
        try:
            path.unlink()
            gone.append(path.name)
        except OSError:
            pass
    return gone


def step_done(r, repo, install):
    started = r.step("done")
    link_boot_romfs(r, repo, install)
    if not r.dry:
        gone = sweep_leftovers(install)
        if gone:
            r.log("  removed after the build: " + ", ".join(sorted(gone)))
        (install / "setup_state.tsv").write_text("mprs_setup_state\t1\nmeta\tfinished\t1\n", encoding="utf-8")
    r.write_status(state="done", note="ready to play")
    return r.end_step("done", started, True, "ready to play")


def main():
    if "--emit-env" in sys.argv[1:]:
        # Prints GAME_ENV as KEY=VALUE lines. tools/run/play.ps1 sources this before launching, so a dev run gets the
        # same switches a package gets from launch.env - the profile no longer keeps a copy that can drift (2026-11-26:
        # a missing NXRT_NVN_EARLYZ_FS=auto turned distant geometry black and looked like a runtime regression).
        for key in sorted(GAME_ENV):
            print("%s=%s" % (key, GAME_ENV[key]))
        return 0
    ap = argparse.ArgumentParser(description="first run of the XC2 NS2 recompilation")
    ap.add_argument("--repo", required=True)
    ap.add_argument("--install", required=True)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--nup")
    g.add_argument("--dump")
    g.add_argument("--exefs", help="a folder that already holds the modules (romfs given with --romfs)")
    ap.add_argument("--romfs", help="the romfs folder when it is not next to the dump")
    ap.add_argument("--jobs", type=int, default=0,
                    help="parallel compiles for the lift; 0 keeps the build script's own default")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--skip", default="", help="comma separated steps to skip")
    a = ap.parse_args()

    repo = Path(a.repo).resolve()
    install = Path(a.install).resolve()
    nup = Path(a.nup).resolve() if a.nup else None
    dump = Path(a.dump).resolve() if a.dump else None
    exefs_in = Path(a.exefs).resolve() if a.exefs else None
    romfs_in = Path(a.romfs).resolve() if a.romfs else None
    r = Runner(install, a.dry_run)
    skip = {s.strip() for s in a.skip.split(",") if s.strip()}
    r.log("repo    %s" % repo)
    r.log("install %s" % install)
    r.log("input   %s" % (nup or dump or exefs_in))

    # Two shapes of the same tree. A package carries runtime/xc2_rt.dll and tools/{python,build/zig}, so nothing is
    # built here and no toolchain has to exist; a checkout builds the runtime with cmake and clang-cl.
    tools = package_tools(repo)
    if tools:
        r.log("mode    package: the runtime and the compilers travel with it")
        steps = [
            ("doctor", lambda: step_doctor_package(r, repo, tools)),
            ("ingest", lambda: step_ingest(r, repo, install, nup, dump, exefs_in, romfs_in)),
            ("lift", lambda: step_lift_package(r, repo, install, tools[0], a.jobs)),
            ("link", lambda: step_link_package(r, repo, install, tools[1], tools[2], a.jobs)),
            ("shaders", lambda: step_shaders_package(r, repo, install, tools[0], a.jobs)),
            ("done", lambda: step_done_package(r, repo, install)),
        ]
    else:
        r.log("mode    checkout: the runtime is built here")
        import_toolchain(r)
        steps = [
            ("doctor", lambda: step_doctor(r, repo)),
            ("sources", lambda: step_sources(r, repo)),
            ("runtime", lambda: step_runtime(r, repo)),
            ("ingest", lambda: step_ingest(r, repo, install, nup, dump, exefs_in, romfs_in)),
            ("lift", lambda: step_lift(r, repo, install, a.jobs)),
            ("link", lambda: step_link(r, repo, install)),
            ("done", lambda: step_done(r, repo, install)),
        ]
    r.total_steps = len(steps)
    r.announce()
    for name, fn in steps:
        if name in skip:
            r.end_step(name, time.time(), True, "skipped")
            continue
        if not fn():
            r.write_status(state="error")
            r.log("stopped at %s" % name)
            return 1
    r.write_status(state="done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
