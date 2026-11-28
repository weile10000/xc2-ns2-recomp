#!/usr/bin/env python3
"""Translate the extracted shader blocks and install them into the cache.

    python tools/shader/cache/build_cache.py --blocks <dir> --cache <dir> --harness <spirv_harness>

This is the second half of the pipeline. The first half is:

    ard_index.py --ard <romfs>/bf2.ard --out files.tsv
    extract_embedded.py --ard <romfs>/bf2.ard --tsv files.tsv --out blocks

and then this script takes the blocks, translates the ones the cache is missing, installs the .spv files and
their resource rows, and (optionally) rebuilds the preload pack.

Design notes, because this is meant to be driven by something other than a person:

  * no prompts, no environment assumptions: every path is an argument;
  * --json emits one object per step, so a shell (a GUI, a wrapper script) can show progress without
    parsing prose;
  * exit codes: 0 ok, 2 bad arguments, 3 input does not match, 4 a dependency is missing, 5 a step failed;
  * idempotent: blocks already in the cache are skipped, so re-running after an interruption continues.
"""
import argparse
import collections
import json
import os
import shutil
import subprocess
import sys

EXIT_OK = 0
EXIT_ARGS = 2
EXIT_INPUT = 3
EXIT_DEP = 4
EXIT_STEP = 5


class Reporter:
    """Prose by default, one JSON object per event with --json."""

    def __init__(self, as_json):
        self.as_json = as_json

    def step(self, name, **fields):
        if self.as_json:
            fields["event"] = "step"
            fields["step"] = name
            print(json.dumps(fields, ensure_ascii=False), flush=True)
        else:
            extra = "  " + " ".join("%s=%s" % (k, v) for k, v in fields.items()) if fields else ""
            print("[%s]%s" % (name, extra), flush=True)

    def say(self, text):
        if not self.as_json:
            print("      " + text, flush=True)

    def done(self, **fields):
        if self.as_json:
            fields["event"] = "done"
            print(json.dumps(fields, ensure_ascii=False), flush=True)
        else:
            print("", flush=True)
            for k, v in fields.items():
                print("  %-18s %s" % (k + ":", v), flush=True)

    def fail(self, code, message):
        if self.as_json:
            print(json.dumps({"event": "error", "code": code, "message": message}, ensure_ascii=False), flush=True)
        else:
            print("ERROR: " + message, file=sys.stderr, flush=True)


def read_manifest(path):
    """The blocks manifest: one row per distinct block, with the hash in column code_sha1_16."""
    rows = []
    with open(path, encoding="utf-8") as f:
        header = f.readline().rstrip("\n").split("\t")
        try:
            i_hash = header.index("code_sha1_16")
            i_code = header.index("code_file")
            i_ctrl = header.index("ctrl_file")
        except ValueError:
            raise SystemExit("the manifest has no code_sha1_16/code_file/ctrl_file column: " + path)
        for line in f:
            raw = line.rstrip("\n")
            p = raw.split("\t")
            if len(p) > max(i_hash, i_code, i_ctrl):
                rows.append({"hash": p[i_hash], "code": p[i_code], "ctrl": p[i_ctrl], "cols": p, "raw": raw})
    return header, rows, (i_code, i_ctrl)


def run_stream(cmd, rep):
    """Run a child command, streaming progress lines live when not emitting JSON."""
    tail = collections.deque(maxlen=40)
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         text=True, errors="replace", bufsize=1)
    for line in p.stdout:
        s = line.rstrip("\r\n")
        tail.append(s)
        if not rep.as_json and "/" in s:
            stripped = s.strip()
            if stripped[:1].isdigit():
                print("      " + stripped, flush=True)
    rc = p.wait()
    return rc, "\n".join(tail)


def move_or_copy(src, dst):
    """Prefer an O(1) same-volume replace over copying bytes and NTFS metadata."""
    try:
        os.replace(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def cached(path):
    if not os.path.isdir(path):
        return set()
    return set(os.path.splitext(n)[0] for n in os.listdir(path) if n.endswith(".spv"))


def main(argv):
    ap = argparse.ArgumentParser(
        description="Translate extracted shader blocks and install them. "
                    "With --ard it runs the whole chain: index, extract, translate, install, pack.")
    ap.add_argument("--ard", default="", help="bf2.ard: run the index and extraction stages first")
    ap.add_argument("--work", default="", help="scratch folder for the chain (default: next to --cache)")
    ap.add_argument("--blocks", default="", help="folder written by extract_embedded.py")
    ap.add_argument("--cache", required=True, help="the spirv cache folder (xc2/build/boot/shader_cache/spirv)")
    ap.add_argument("--harness", required=True, help="spirv_harness executable")
    ap.add_argument("--spirv-val", default="", help="spirv-val executable the harness validates with; unset: next to the "
                                                   "harness in tools/spirv-val, else the harness's own lookup (VULKAN_SDK, PATH)")
    ap.add_argument("--no-val", action="store_true", help="skip spirv-val per-shader subprocess validation")
    ap.add_argument("--refresh", action="store_true",
                    help="re-translate archive blocks and refresh their .spv/results.tsv entries in-place without deleting other cached shaders (append-only)")
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--stage", default="", help="scratch folder (default: <blocks>/../build_cache)")
    ap.add_argument("--pack-tool", default="", help="optional pack_full.py; without it the pack is left alone")
    ap.add_argument("--pack", default="", help="the spirv.preload.pack to rebuild (needs --pack-tool)")
    ap.add_argument("--programs", default="", help="output MPRPRG01 programs.bin path (default with --ard: <cache>/../programs.bin)")
    ap.add_argument("--dry-run", action="store_true", help="report what would be translated, change nothing")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    rep = Reporter(a.json)

    here = os.path.dirname(os.path.abspath(__file__))
    work = a.work or (os.path.join(os.path.dirname(os.path.abspath(a.cache)), "shader-cache-work"))
    if a.ard:
        if not os.path.exists(a.ard):
            rep.fail(EXIT_INPUT, "no archive at " + a.ard)
            return EXIT_INPUT
        os.makedirs(work, exist_ok=True)
        tsv = os.path.join(work, "files.tsv")
        rep.step("index", ard=a.ard)
        rc, tail = run_stream([sys.executable, os.path.join(here, "ard_index.py"),
                               "--ard", a.ard, "--out", tsv], rep)
        if rc != 0:
            rep.say(tail[-600:])
            rep.fail(EXIT_INPUT, "the index step failed")
            return EXIT_INPUT
        a.blocks = os.path.join(work, "blocks")
        programs_out = a.programs or os.path.join(os.path.dirname(os.path.abspath(a.cache)), "programs.bin")
        rep.step("extract", ard=a.ard, out=a.blocks, jobs=a.jobs)
        rc, tail = run_stream([sys.executable, os.path.join(here, "extract_embedded.py"),
                               "--ard", a.ard, "--tsv", tsv, "--out", a.blocks,
                               "--programs", programs_out, "--jobs", str(a.jobs)], rep)
        if rc != 0:
            rep.say(tail[-600:])
            rep.fail(EXIT_INPUT, "the extraction step failed")
            return EXIT_INPUT
    if not a.blocks:
        rep.fail(EXIT_ARGS, "give --blocks, or --ard to run the whole chain")
        return EXIT_ARGS

    manifest = os.path.join(a.blocks, "manifest.tsv")
    if not os.path.exists(manifest):
        rep.fail(EXIT_INPUT, "no manifest.tsv in " + a.blocks)
        return EXIT_INPUT
    if not os.path.exists(a.harness):
        rep.fail(EXIT_DEP, "no shader harness at " + a.harness)
        return EXIT_DEP

    header, rows, (i_code, i_ctrl) = read_manifest(manifest)
    if not rows:
        rep.fail(EXIT_INPUT, "the manifest has no rows")
        return EXIT_INPUT
    have = cached(a.cache)
    missing = list(rows) if a.refresh else [r for r in rows if r["hash"] not in have]
    rep.step("plan", blocks=len(rows), cached=len(have), missing=len(missing), refresh=a.refresh)
    if not missing:
        rep.done(translated=0, installed=0, note="the cache already covers every block")
        return EXIT_OK
    if a.dry_run:
        rep.done(translated=0, installed=0, would_translate=len(missing), note="dry run")
        return EXIT_OK

    stage = a.stage or os.path.join(work, "build_cache")
    stage_in = os.path.join(stage, "in")
    stage_out = os.path.join(stage, "out")
    if os.path.isdir(stage_out):
        shutil.rmtree(stage_out)
    os.makedirs(stage_out)

    blocks_abs = os.path.abspath(a.blocks)
    out_abs = os.path.abspath(stage_out)
    direct = (len(missing) == len(rows)) and not out_abs.startswith(blocks_abs + os.sep)
    if direct:
        if not a.ard:
            for r in missing:
                for src in (os.path.join(a.blocks, r["code"]), os.path.join(a.blocks, r["ctrl"])):
                    if not os.path.exists(src):
                        rep.fail(EXIT_INPUT, "the manifest names a file that is not there: " + src)
                        return EXIT_INPUT
        rep.step("stage", folder=a.blocks, mode="direct")
        batch_in = a.blocks
    else:
        if os.path.isdir(stage_in):
            shutil.rmtree(stage_in)
        os.makedirs(stage_in)
        rep.step("stage", folder=stage_in, mode="manifest")
        with open(os.path.join(stage_in, "manifest.tsv"), "w", encoding="utf-8", newline="\n") as f:
            f.write("\t".join(header) + "\n")
            for r in missing:
                code_src = os.path.abspath(os.path.join(a.blocks, r["code"]))
                ctrl_src = os.path.abspath(os.path.join(a.blocks, r["ctrl"]))
                for src in (code_src, ctrl_src):
                    if not os.path.exists(src):
                        rep.fail(EXIT_INPUT, "the manifest names a file that is not there: " + src)
                        return EXIT_INPUT
                cols = list(r["cols"])
                cols[i_code] = code_src
                cols[i_ctrl] = ctrl_src
                f.write("\t".join(cols) + "\n")
        batch_in = stage_in

    rep.step("translate", count=len(missing), jobs=a.jobs)
    cmd = [a.harness, "--batch", batch_in, "--out", stage_out, "--jobs", str(a.jobs)]
    if a.no_val:
        cmd.append("--no-val")
    else:
        # The harness validates every translation with spirv-val, and a machine without the Vulkan SDK has none: it used
        # to die with an unhandled Win32Exception in the middle of the batch (issue #5, 2026-11-18). The release package
        # ships it in tools/spirv-val, next to the harness's own folder.
        spirv_val = a.spirv_val
        if not spirv_val:
            guess = os.path.join(os.path.dirname(os.path.abspath(a.harness)), "..", "spirv-val", "spirv-val.exe")
            guess = os.path.normpath(guess)
            if os.path.exists(guess):
                spirv_val = guess
        if spirv_val:
            if not os.path.exists(spirv_val):
                rep.fail(EXIT_DEP, "no spirv-val at " + spirv_val)
                return EXIT_DEP
            cmd += ["--spirv-val", spirv_val]
    rc, tail = run_stream(cmd, rep)
    if rc != 0:
        rep.say(tail[-800:])
        rep.fail(EXIT_STEP, "the shader harness failed with exit %d" % rc)
        return EXIT_STEP

    rep.step("install", cache=a.cache)
    os.makedirs(a.cache, exist_ok=True)
    spv_files = [n for n in sorted(os.listdir(stage_out)) if n.endswith(".spv")]
    # Only the rows of shaders that actually produced a .spv: the harness writes a row for a block it could
    # not translate too, and a row without a .spv would make the runtime resolve a shader that is not there.
    produced = set(os.path.splitext(n)[0] for n in spv_files)
    copied = 0
    for n in spv_files:
        move_or_copy(os.path.join(stage_out, n), os.path.join(a.cache, n))
        copied += 1
    added = 0
    results = os.path.join(a.cache, "results.tsv")
    new_rows = os.path.join(stage_out, "results.tsv")
    if os.path.exists(new_rows):
        # The header has to come from the harness, not from us: the runtime finds the resource records by
        # looking for a column literally named "resources" (parse_results_tsv in nvn_shaders.cpp), and a cache
        # written under any other header silently loses every shader's descriptor bindings.
        known = set()
        with open(new_rows, encoding="utf-8") as f:
            head = f.readline().rstrip("\n")
        if "resources" not in head.split("\t"):
            rep.fail(EXIT_INPUT, "the harness output has no 'resources' column: " + head)
            return EXIT_INPUT
        if a.refresh and os.path.exists(results):
            # Append-only refresh: update rows for newly produced shaders in place while preserving every other
            # cached row (such as the 5 runtime-generated prefill shaders that never appear in bf2.ard).
            updated = {}
            with open(new_rows, encoding="utf-8") as f:
                f.readline()
                for line in f:
                    line = line.rstrip("\n")
                    if line:
                        h = line.split("\t", 1)[0]
                        if h in produced:
                            updated[h] = line
            if updated:
                bak = results + ".bak-before-build"
                if not os.path.exists(bak):
                    shutil.copy2(results, bak)
                merged = []
                seen = set()
                with open(results, encoding="utf-8") as f:
                    old_head = f.readline().rstrip("\n")
                    if old_head != head:
                        rep.say("note: the cache was written under a different header")
                        rep.say("      cache : " + old_head)
                        rep.say("      harness: " + head)
                    for line in f:
                        raw = line.rstrip("\n")
                        if not raw.strip():
                            continue
                        h = raw.split("\t", 1)[0]
                        if h in updated:
                            merged.append(updated[h])
                            seen.add(h)
                        else:
                            merged.append(raw)
                            seen.add(h)
                for h, line in updated.items():
                    if h not in seen:
                        merged.append(line)
                tmp_results = results + ".tmp"
                with open(tmp_results, "w", encoding="utf-8", newline="\n") as f:
                    f.write(head + "\n")
                    for line in merged:
                        f.write(line + "\n")
                os.replace(tmp_results, results)
                added = len(updated)
        else:
            if os.path.exists(results):
                with open(results, encoding="utf-8") as f:
                    old_head = f.readline().rstrip("\n")
                    if old_head != head:
                        rep.say("note: the cache was written under a different header")
                        rep.say("      cache : " + old_head)
                        rep.say("      harness: " + head)
                    for line in f:
                        if line.strip():
                            known.add(line.split("\t", 1)[0])
            add = []
            with open(new_rows, encoding="utf-8") as f:
                f.readline()
                for line in f:
                    line = line.rstrip("\n")
                    if line and line.split("\t", 1)[0] not in known and line.split("\t", 1)[0] in produced:
                        add.append(line)
            if add:
                if os.path.exists(results):
                    bak = results + ".bak-before-build"
                    if not os.path.exists(bak):
                        shutil.copy2(results, bak)
                else:
                    with open(results, "w", encoding="utf-8", newline="\n") as f:
                        f.write(head + "\n")
                with open(results, "a", encoding="utf-8", newline="\n") as f:
                    for line in add:
                        f.write(line + "\n")
                added = len(add)

    packed = "left alone"
    pack_tool = a.pack_tool or os.path.join(here, "pack.py")
    if a.pack and not os.path.exists(pack_tool):
        rep.fail(EXIT_DEP, "asked to rebuild the pack but there is no pack tool at " + pack_tool)
        return EXIT_DEP
    if a.pack:
        rep.step("pack", tool=os.path.basename(pack_tool))
        cmd = [sys.executable, pack_tool, "--rows", results, "--cache", a.cache,
               "--out", os.path.join(stage, "spirv.preload.pack"), "--install", a.pack]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            rep.say((r.stderr or r.stdout or "").strip()[-800:])
            rep.fail(EXIT_STEP, "the pack tool failed with exit %d" % r.returncode)
            return EXIT_STEP
        packed = a.pack
    elif a.refresh and copied > 0:
        # spirv.preload.pack is read before loose .spv files and hashes results.tsv rows rather than .spv bytes;
        # explicitly drop any stale pack after refreshing shaders so the runtime rebuilds it from the fresh .spv files.
        dropped = []
        cache_abs = os.path.abspath(a.cache)
        for cand in (os.path.join(os.path.dirname(cache_abs), "spirv.preload.pack"),
                     os.path.join(cache_abs, "spirv.preload.pack")):
            if os.path.isfile(cand):
                try:
                    os.unlink(cand)
                    dropped.append(cand)
                except OSError:
                    pass
        if dropped:
            packed = "dropped stale pack"
            rep.say("dropped stale preload pack: " + ", ".join(dropped))

    after = cached(a.cache)
    # A block the harness could not translate is not a failure of this script: on the retail archive 22 of the
    # 12,126 extracted blocks are misaligned garbage whose hashes the game never asks for. Report them, and only
    # fail when nothing at all came out.
    rejected = sorted(r["hash"] for r in missing if r["hash"] not in after)
    rep.done(translated=len(missing), installed=copied, results_rows=added, pack=packed,
             cache_entries=len(after), rejected=len(rejected))
    if rejected:
        rep.say("rejected by the translator (their hashes are not ones the game requests):")
        for h in rejected[:10]:
            rep.say("  " + h)
        if len(rejected) > 10:
            rep.say("  ... and %d more" % (len(rejected) - 10))
    if copied == 0:
        rep.fail(EXIT_STEP, "not one shader came out of the translator")
        return EXIT_STEP
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
