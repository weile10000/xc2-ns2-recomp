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
            p = subprocess.Popen([str(a) for a in argv], cwd=str(cwd) if cwd else None, env=full_env,
                                 stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                 text=True, errors="replace", bufsize=1)
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


def pwsh(*args):
    return ["pwsh", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", *args]


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
        p = subprocess.run(["pwsh", "-NoProfile", "-Command",
                            ". $env:XC2_ENV *> $null; Write-Output $env:PATH"],
                           capture_output=True, text=True, timeout=180)
    except OSError as exc:
        r.log("  could not run pwsh to read XC2_ENV: %s" % exc)
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


def step_lift(r, repo, install, jobs=0):
    started = r.step("lift")
    lib = repo / "xc2" / "build" / "301" / "lib"
    if all((lib / (m + ".lib")).is_file() for m in MODULES):
        return r.end_step("lift", started, True, "the libraries are already there")
    argv = pwsh(repo / "tools" / "build" / "build.ps1", "-Exefs", install / "exefs", "-Run:$false",
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
    if exe.is_file():
        return r.end_step("link", started, True, "the boot exe is already there")
    rc = r.run(pwsh(repo / "tools" / "build" / "build-boot.ps1", "-Exefs", install / "exefs",
                    "-OutDir", repo / "xc2" / "build" / "301" / "boot", "-LibDir", repo / "xc2" / "build" / "301" / "lib",
                    "-MainLib", repo / "xc2" / "build" / "301" / "lib" / "main.lib", "-Defines", "XC2_MODULES_7"), cwd=repo,
                on_line=build_progress(r, "link"))
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
ZIG_FLAGS = ["-target", "x86_64-windows-gnu", "-std=c++20", "-O1", "-w", "-fno-exceptions", "-fno-rtti",
             "-fno-strict-aliasing", "-ffp-contract=off", "-fno-math-errno", "-fno-threadsafe-statics",
             "-DNX_INLINE_INDIRECT"]


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
    try:
        out = subprocess.run([str(zig), "targets"], capture_output=True, text=True, errors="replace", timeout=120)
        text = out.stdout
        i = text.find(".native")
        j = text.find(".features", i)
        if i < 0 or j < 0:
            return "x86_64+cx16+fma"
        feats = re.findall(r'"([^"]+)"', text[j:text.find("}", j)])
    except (OSError, subprocess.SubprocessError):
        return "x86_64+cx16+fma"
    need = ("cx16", "sse4_2", "avx", "fma")
    return "x86_64+cx16+fma" if all(f in feats for f in need) else "x86_64+cx16+sse4_2+popcnt"


def step_doctor_package(r, repo, tools):
    started = r.step("doctor")
    py, zig, lib = tools
    rc = r.run([str(py), "-c", "import capstone, lz4.block, zstandard, nxrecomp"])
    if rc != 0:
        return r.end_step("doctor", started, False, "the bundled Python cannot import the lifter (capstone/lz4/zstandard/nxrecomp)")
    r.log("  python %s" % py)
    r.log("  zig    %s (%s)" % (zig, cpu_variant(zig)))
    r.log("  runtime %s" % lib)
    # The harness validates every translated shader with spirv-val; without it the shader step dies halfway through
    # with an unhandled Win32Exception from inside the harness (issue #5, 2026-11-18). The package ships it, and a
    # machine with the Vulkan SDK also has one on PATH, so this is a warning rather than a failure.
    val = repo / "tools" / "spirv-val" / "spirv-val.exe"
    if val.is_file():
        r.log("  spirv-val %s" % val)
    else:
        r.log("  spirv-val not in the package; the harness will look in %%VULKAN_SDK%% and on PATH, "
              "and the shader step fails if neither has it")
    return r.end_step("doctor", started, True, "the package carries everything")


def step_lift_package(r, repo, install, py, jobs):
    started = r.step("lift")
    # Build products live with the game, not in the package: a player's install folder is theirs to fill, and the
    # package stays exactly as it was unpacked.
    lift = install / "build" / "lift"
    sites = repo / "xc2" / "sites"
    n = jobs if jobs > 0 else max(2, (os.cpu_count() or 4) - 1)
    done = 0
    for m in MODULES:
        out = lift / m
        if (out / ("module_%s.cpp" % m)).is_file():
            r.log("  %s: already lifted" % m)
            done += 1
            continue
        argv = [str(py), "-m", "nxrecomp", "lift", str(install / "exefs" / m), str(out),
                "--game", "xc2ns2", "--name", m, "--base", BASES[m], "--jobs", str(n)]
        s = sites / ("sites-%s.txt" % m)
        if s.is_file():
            argv += ["--sites", str(s)]
        if r.run(argv, cwd=repo) != 0:
            return r.end_step("lift", started, False, "the lifter failed on " + m)
        done += 1
        r.note_progress("lift %s (%d/%d)" % (m, done, len(MODULES)), 100.0 * done / len(MODULES))
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


def step_link_package(r, repo, install, zig, rtlib, jobs):
    started = r.step("link")
    exe = install / "xc2_boot.exe"
    if exe.is_file():
        # The exe is kept across an upgrade; the DLLs still have to land next to it (issue #2).
        copy_runtime_dlls(r, rtlib.parent, install)
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
        # One at a time is about 27 minutes for the 537 translation units; the compiler is the only thing using the
        # machine while this runs, so it takes every core but one.
        n = jobs if jobs > 0 else max(2, (os.cpu_count() or 4) - 1)
        r.log("  compiling %d of %d file(s), %d at a time" % (len(todo), len(srcs), n))

        def build(item):
            s, o = item
            p = subprocess.run(flags + ["-I" + str(inc), "-I" + str(s.parent), "-c", str(s), "-o" + str(o)],
                               cwd=str(repo), capture_output=True, text=True, errors="replace")
            return s, p.returncode, (p.stderr or "")[-400:]

        from concurrent.futures import ThreadPoolExecutor, as_completed
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
    argv = ([str(zig), "c++", "-target", "x86_64-windows-gnu", "-O1", "-w", "-fno-exceptions", "-fno-rtti",
             "-DXC2_MODULES_7", "@" + str(rsp), str(stub), "-I" + str(inc), "-I" + str(stub.parent),
             str(rtlib), "-o", str(exe)])
    if r.run(argv) != 0:
        return r.end_step("link", started, False, "the exe did not link")
    copy_runtime_dlls(r, rtlib.parent, install)
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


def find_romfs(dump, explicit):
    # --romfs wins when it is given. Measured 2026-11-17: a run that passed --romfs looked like it had linked the
    # maintainer tree instead, and had not - Path.resolve() dereferences a junction, and the folder passed in WAS a
    # junction to that one, so the install's link named the real target. Both names reach the same files.
    if explicit:
        return explicit if (explicit / "bf2.ard").is_file() else None
    for candidate in (dump, dump / "romfs", dump.parent, dump.parent / "romfs"):
        if (candidate / "bf2.ard").is_file():
            return candidate
    return None


def link_dlc(install, r):
    """Link the dump's add-on content in: without it the game loads no DLC at all.

    The launcher hands the folder over as XC2_AOC_DIR, because it is the side that scanned the dump; the game reads
    <install>/game/dlc-301, so that path gets the junction. A dump without add-ons is not an error.
    """
    src = os.environ.get("XC2_AOC_DIR", "")
    if not src or not Path(src).is_dir():
        r.log("  dlc: none in the dump (the game runs without add-on content)")
        return
    game = install / "game"
    if not r.dry:
        game.mkdir(parents=True, exist_ok=True)
    if junction(game / "dlc-301", Path(src), r) == 0:
        r.log("  dlc: linked %s" % src)


def step_ingest(r, repo, install, nup, dump, exefs_in, romfs_in):
    started = r.step("ingest")
    exefs = install / "exefs"
    romfs = install / "romfs"
    if have_modules(exefs) and romfs.is_dir():
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
        link_dlc(install, r)
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
        link_dlc(install, r)
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
    # The game's language is NOT here on purpose: the player changes it in the launcher, which writes [system] language
    # into nxrt.ini. An env value would win over that file and the setting would do nothing.
    "NXRT_TITLE_SUPPORTED_LANGUAGES": "Japanese,AmericanEnglish,BritishEnglish,French,German,Italian,Spanish,"
                                       "LatinAmericanSpanish,Chinese,Taiwanese,Korean,SimplifiedChinese,TraditionalChinese",
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
    """
    started = r.step("shaders")
    spirv = install / "shader_cache" / "spirv"
    harness = repo / "tools" / "spirv_harness" / "spirv_harness.exe"
    # Two halves, two markers: results.tsv says the archive was walked, "prefilled" says the headless run named what no
    # file contains. They are separate on purpose - a setup that dies between them used to look finished (12,132 in the
    # cache, 5 missing, black title) and every later setup skipped the step entirely.
    scanned = (spirv / "results.tsv").is_file() and any(spirv.glob("*.spv"))
    prefilled = install / "build" / "prefilled"
    if scanned and prefilled.is_file():
        return r.end_step("shaders", started, True, "the shader cache is already there")
    if not scanned:
        tool = first_file(repo / "tools" / "shader" / "cache" / "build_cache.py", repo / "tools" / "shader-cache" / "build_cache.py")
        ard = install / "romfs" / "bf2.ard"
        if not tool.is_file():
            return r.end_step("shaders", started, False, "the package has no tools/shader-cache/build_cache.py")
        if not harness.is_file():
            return r.end_step("shaders", started, False, "the package has no shader harness")
        if not ard.is_file():
            return r.end_step("shaders", started, False, "no bf2.ard under " + str(install / "romfs"))
        n = jobs if jobs > 0 else max(2, (os.cpu_count() or 4) - 1)
        r.log("  reading the archive and translating what the cache lacks; this is the long step")
        # The progress filter drops everything that is not a stage line, which also dropped the child's traceback:
        # a failing shader step used to say "the shader chain failed" and nothing else (issue #5, 2026-11-18). Keep the
        # tail and print it when the child exits non-zero.
        tail = collections.deque(maxlen=40)
        progress = shader_progress(r)

        def keep(line, tail=tail, progress=progress):
            tail.append(line)
            progress(line)

        rc = r.run([str(py), str(tool), "--ard", str(ard), "--cache", str(spirv), "--harness", str(harness),
                    "--work", str(install / "build" / "shader-cache"), "--jobs", str(n)],
                   cwd=repo, on_line=keep)
        if rc != 0:
            r.log("  the shader chain exited with code %d; its last lines:" % rc)
            for ln in tail:
                r.log("  | " + ln.rstrip()[:300])
            return r.end_step("shaders", started, False, "the shader chain failed (exit %d); see the lines above" % rc)
        r.log("  %d shader(s) from the archive" % len(list(spirv.glob("*.spv"))))
    else:
        # Re-walking the archive only retries the 22 blocks the translator rejects, and build_cache.py treats "not one
        # shader came out" as a hard failure - measured 2026-11-18. The scan half is done; only the prefill is missing.
        r.log("  the archive scan is already there; only the headless prefill is missing")
    note = prefill_package(r, repo, install, py, harness, spirv, jobs=jobs)
    r.log("  %d shader(s) in the cache" % len(list(spirv.glob("*.spv"))))
    if not prefilled.is_file():
        return r.end_step("shaders", started, False,
                          note or "the headless prefill named no block; the cache has the archive's share only")
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
    argv = [str(exe), "--exefs", str(install / "exefs"), "--link", "host", "--gfx", "headless", "--log", str(log)]
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
    deadline = time.time() + seconds
    seen = 0
    while time.time() < deadline and p.poll() is None:
        n = len(list(missing.glob("code_*.bin")))
        if n != seen:
            seen = n
            r.note_progress("prefill: %d block(s) collected" % n)
        time.sleep(1)
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
            return "the headless prefill could not start: " + (short or reason[:96] or hint[:96])
        retype_ds_samplers(r, repo, py, spirv, log)
        prefilled.write_text("0 collected: nothing was missing\n", encoding="utf-8")
        return ""
    n = jobs if jobs > 0 else max(2, (os.cpu_count() or 4) - 1)
    rc = r.run([str(py), str(first_file(repo / "tools" / "shader" / "cache" / "build_cache.py", repo / "tools" / "shader-cache" / "build_cache.py")), "--blocks", str(missing),
                "--cache", str(spirv), "--harness", str(harness), "--jobs", str(n)], cwd=repo,
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
    """build_cache.py reports each stage on its own line; pass those on so the page shows movement."""
    def on_line(line):
        s = line.strip()
        if s[:1] == "[" and ("plan" in s or "translate" in s or "install" in s or "pack" in s):
            r.note_progress(s[:140])
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
    game = install / "game"
    if not r.dry:
        game.mkdir(parents=True, exist_ok=True)
    junction(game / "romfs-301", install / "romfs", r)
    if (install / "nup_dump" / "dlc").is_dir():
        junction(game / "dlc-301", install / "nup_dump" / "dlc", r)
    if not r.dry:
        (install / "setup_state.tsv").write_text("mprs_setup_state\t1\nmeta\tfinished\t1\n", encoding="utf-8")
    r.write_status(state="done", note="ready to play")
    return r.end_step("done", started, True, "ready to play")


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
    for cand in (install / "nup_dump" / "dlc", install / "game" / "dlc-301", install / "dlc"):
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
