#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Unit tests for build/install-manifest.json, legacy seeding, ABI decoupling, and append-only shader refresh."""
import importlib.util
import json
import pathlib
import stat
import struct
import subprocess
import sys
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[2]


def load_setup():
    spec = importlib.util.spec_from_file_location("xc2_setup", ROOT / "tools" / "setup" / "xc2_setup.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_minimal_pe64_dll(export_names, extra_padding=0):
    """Build a tiny, valid PE32+ DLL in memory carrying the given sorted export names."""
    names = sorted(export_names)
    sec_rva = 0x1000
    sec_off = 0x200
    exp_dir_sz = 40
    addr_tbl_off = exp_dir_sz
    name_ptr_off = addr_tbl_off + 4 * len(names)
    ord_tbl_off = name_ptr_off + 4 * len(names)
    str_off = ord_tbl_off + 2 * len(names)

    str_blob = bytearray()
    name_rvas = []
    for n in names:
        name_rvas.append(sec_rva + str_off + len(str_blob))
        str_blob.extend(n.encode("ascii") + b"\0")
    dll_name_rva = sec_rva + str_off + len(str_blob)
    str_blob.extend(b"xc2_rt.dll\0")

    sec_data = bytearray(str_off)
    struct.pack_into(
        "<IIIIIIIIII",
        sec_data,
        0,
        0, 0, 0,
        dll_name_rva,
        1,
        len(names),
        len(names),
        sec_rva + addr_tbl_off,
        sec_rva + name_ptr_off,
        sec_rva + ord_tbl_off,
    )
    for i, nrva in enumerate(name_rvas):
        struct.pack_into("<I", sec_data, addr_tbl_off + i * 4, sec_rva + 0x10 + i)
        struct.pack_into("<I", sec_data, name_ptr_off + i * 4, nrva)
        struct.pack_into("<H", sec_data, ord_tbl_off + i * 2, i)
    sec_data.extend(str_blob)
    if extra_padding:
        sec_data.extend(b"\x90" * extra_padding)
    raw_sz = ((len(sec_data) + 0x1FF) // 0x200) * 0x200
    sec_data.extend(b"\0" * (raw_sz - len(sec_data)))

    hdr = bytearray(sec_off)
    hdr[0:2] = b"MZ"
    pe_off = 0x80
    struct.pack_into("<I", hdr, 0x3C, pe_off)
    hdr[pe_off : pe_off + 4] = b"PE\0\0"
    opt_sz = 0xF0
    struct.pack_into("<HHIIIHH", hdr, pe_off + 4, 0x8664, 1, 0, 0, 0, opt_sz, 0x2022)
    opt_off = pe_off + 24
    struct.pack_into("<H", hdr, opt_off, 0x20B)  # PE32+
    struct.pack_into("<I", hdr, opt_off + 108, 16)  # NumberOfRvaAndSizes
    struct.pack_into("<II", hdr, opt_off + 112, sec_rva, len(sec_data))  # Export Directory
    sec_hdr = opt_off + opt_sz
    hdr[sec_hdr : sec_hdr + 8] = b".edata\0\0"
    struct.pack_into("<IIII", hdr, sec_hdr + 8, len(sec_data), sec_rva, raw_sz, sec_off)
    return bytes(hdr) + bytes(sec_data)


class FakeRunner:
    def __init__(self, dry=False):
        self.dry = dry
        self.logs = []
        self.cmds = []
        self.ended = None

    def step(self, name):
        return 0.0

    def log(self, msg):
        self.logs.append(msg)

    def note_progress(self, detail, percent=None):
        pass

    def run(self, argv, cwd=None, env=None, on_line=None):
        self.cmds.append([str(a) for a in argv])
        return 0

    def end_step(self, name, started, ok, note=""):
        self.ended = (name, ok, note)
        return ok


class InstallManifestTests(unittest.TestCase):
    def setUp(self):
        self.setup = load_setup()
        self.td = tempfile.TemporaryDirectory()
        self.base = pathlib.Path(self.td.name)
        self.repo = self.base / "pkg"
        self.install = self.base / "install"
        (self.repo / "xc2" / "sites").mkdir(parents=True)
        (self.repo / "xc2" / "src").mkdir(parents=True)
        (self.repo / "runtime").mkdir(parents=True)
        (self.repo / "tools" / "spirv_harness").mkdir(parents=True)
        (self.repo / "tools" / "shader-cache").mkdir(parents=True)
        (self.repo / "tools" / "python" / "Lib" / "site-packages" / "nxrecomp").mkdir(parents=True)
        (self.install / "exefs").mkdir(parents=True)
        (self.install / "romfs").mkdir(parents=True)
        (self.install / "build").mkdir(parents=True)

        (self.repo / "xc2" / "src" / "xc2_stub.cpp").write_text("int main(){return 0;}\n", encoding="utf-8")
        (self.repo / "xc2" / "src" / "xc2_modules.h").write_text("// mods\n", encoding="utf-8")
        (self.repo / "xc2" / "sites" / "sites-main.txt").write_text("0x100\n", encoding="utf-8")
        (self.repo / "tools" / "python" / "Lib" / "site-packages" / "nxrecomp" / "program.py").write_text("# v1\n", encoding="utf-8")
        (self.repo / "tools" / "spirv_harness" / "spirv_harness.exe").write_bytes(b"apphost")
        (self.repo / "tools" / "spirv_harness" / "Ryujinx.Graphics.Shader.dll").write_bytes(b"translator-v1")
        (self.repo / "tools" / "spirv_harness" / "Spv.Generator.dll").write_bytes(b"spvgen-v1")
        (self.repo / "tools" / "spirv_harness" / "AmpereDecoder.dll").write_bytes(b"ampere-v1")
        (self.repo / "tools" / "spirv_harness" / "Gommon.dll").write_bytes(b"gommon-v1")
        (self.repo / "tools" / "spirv_harness" / "spirv_harness.pdb").write_bytes(b"pdb-v1")
        (self.repo / "tools" / "shader-cache" / "build_cache.py").write_text("# cache v1\n", encoding="utf-8")
        (self.install / "romfs" / "bf2.ard").write_bytes(b"ard-data-v1")
        for m in self.setup.MODULES:
            (self.install / "exefs" / m).write_bytes(b"NSO0" + m.encode("ascii"))

    def tearDown(self):
        self.td.cleanup()

    def _seed_lifted(self):
        for m in self.setup.MODULES:
            d = self.install / "build" / "lift" / m
            d.mkdir(parents=True, exist_ok=True)
            (d / ("module_%s.cpp" % m)).write_text("// lifted\n", encoding="utf-8")

    def test_legacy_lift_seeding_skips_relift(self):
        self._seed_lifted()
        r = FakeRunner()
        ok = self.setup.step_lift_package(r, self.repo, self.install, pathlib.Path("python"), jobs=2)
        self.assertTrue(ok)
        self.assertEqual(r.cmds, [], "legacy install with existing lift must not re-lift")
        m = self.setup.load_install_manifest(self.install)
        self.assertIn("lift", m)
        self.assertEqual(set(m["lift"]["modules"].keys()), set(self.setup.MODULES))

    def test_sites_change_re_lifts_only_affected_module(self):
        self._seed_lifted()
        r1 = FakeRunner()
        self.assertTrue(self.setup.step_lift_package(r1, self.repo, self.install, pathlib.Path("python"), jobs=2))
        self.assertEqual(r1.cmds, [])

        # Update only sites-main.txt: only main should be wiped and queued for lift.
        (self.repo / "xc2" / "sites" / "sites-main.txt").write_text("0x100\n0x200\n", encoding="utf-8")
        r2 = FakeRunner()
        self.assertTrue(self.setup.step_lift_package(r2, self.repo, self.install, pathlib.Path("python"), jobs=2))
        self.assertEqual(len(r2.cmds), 1)
        self.assertFalse((self.install / "build" / "lift" / "main" / "module_main.cpp").exists())
        for other in self.setup.MODULES:
            if other != "main":
                self.assertTrue((self.install / "build" / "lift" / other / ("module_%s.cpp" % other)).is_file())

    def test_pe_exports_abi_decoupling_skips_relink_and_copies_dlls(self):
        exports_v1 = [
            "xc2_rt_main", "xc2_rt_set_modules", "call_indirect", "host_import",
            "guest_trap", "guest_unimpl", "guest_svc", "guest_mrs", "guest_msr",
            "nx_site_inline", "nx_site_midhook", "nx_hook_entry", "stx_diag",
            "nx_catch_call", "nx_fn_flat", "nx_fn_flat_n", "nx_fn_flat_host",
        ]
        dll_path = self.repo / "runtime" / "xc2_rt.dll"
        lib_path = self.repo / "runtime" / "xc2_rt.lib"
        ff_dll = self.repo / "runtime" / "avcodec-63.dll"
        dll_v1 = make_minimal_pe64_dll(exports_v1, extra_padding=0)
        dll_path.write_bytes(dll_v1)
        lib_path.write_bytes(b"lib-v1")
        ff_dll.write_bytes(b"ff-v1")
        self.assertEqual(self.setup.pe_export_names(dll_path), sorted(exports_v1))

        # Seed an already-built install with manifest matching v1.
        exe = self.install / "xc2_boot.exe"
        exe.write_bytes(b"MZ" + b"\0" * 128)
        orig_cpu = self.setup.cpu_variant
        self.setup.cpu_variant = lambda _zig: "x86_64+cx16+fma"
        try:
            want_v1 = self.setup.link_inputs_hash(self.repo, self.install, lib_path, None, "x86_64+cx16+fma")
            self.setup.save_install_manifest(self.install, {"link": want_v1})

            # Upgrade xc2_rt.dll with DIFFERENT size/bytes but the EXACT SAME 17 exports (routine renderer fix).
            dll_v2 = make_minimal_pe64_dll(exports_v1, extra_padding=1024)
            self.assertNotEqual(len(dll_v1), len(dll_v2))
            dll_path.write_bytes(dll_v2)
            lib_path.write_bytes(b"lib-v2-different-size")
            ff_dll.write_bytes(b"ff-v2")

            r = FakeRunner()
            ok = self.setup.step_link_package(r, self.repo, self.install, pathlib.Path("zig"), lib_path, jobs=2)
            self.assertTrue(ok)
            self.assertEqual(r.cmds, [], "same-ABI runtime update must skip Zig relink")
            self.assertEqual((self.install / "xc2_rt.dll").read_bytes(), dll_v2)
            self.assertEqual((self.install / "avcodec-63.dll").read_bytes(), b"ff-v2")

            # Now add an 18th export -> ABI changed -> must invoke Zig link!
            dll_v3 = make_minimal_pe64_dll(exports_v1 + ["xc2_rt_new_export"])
            dll_path.write_bytes(dll_v3)
            self._seed_lifted()
            r2 = FakeRunner()
            # Pre-create .o files newer than .cpp so only the link command runs via FakeRunner.
            for m in self.setup.MODULES:
                od = self.install / "build" / "obj-zig" / m
                od.mkdir(parents=True, exist_ok=True)
                (od / ("module_%s.o" % m)).write_bytes(b"obj")
            ok2 = self.setup.step_link_package(r2, self.repo, self.install, pathlib.Path("zig"), lib_path, jobs=2)
            self.assertTrue(ok2)
            self.assertEqual(len(r2.cmds), 1, "changed PE export table must trigger Zig relink")
        finally:
            self.setup.cpu_variant = orig_cpu

    def test_legacy_shaders_seeding_and_runtime_only_upgrade_skip_retranslate_and_prefill(self):
        spirv = self.install / "shader_cache" / "spirv"
        spirv.mkdir(parents=True)
        (spirv / "results.tsv").write_text("hash\tresources\n0001\tr\n", encoding="utf-8")
        (spirv / "0001.spv").write_bytes(b"\x03\x02\x23\x07")
        (self.install / "build" / "prefilled").write_text("5 collected\n", encoding="utf-8")
        (self.install / "runtime").mkdir(parents=True)
        (self.install / "runtime" / "xc2_rt.dll").write_bytes(b"rt-v1")

        prefill_calls = []
        orig_prefill = self.setup.prefill_package
        self.setup.prefill_package = lambda *a, **kw: (prefill_calls.append(1) or "")
        try:
            # 1) Legacy install with no manifest and no prefill-stamp: seeds without translating or prefilling.
            r1 = FakeRunner()
            self.assertTrue(self.setup.step_shaders_package(r1, self.repo, self.install, pathlib.Path("python"), jobs=2))
            self.assertEqual(r1.cmds, [])
            self.assertEqual(prefill_calls, [])
            m = self.setup.load_install_manifest(self.install)
            self.assertIn("shaders", m)
            self.assertIn("prefill", m)

            # 2) Subsequent runtime-only DLL update (translator unchanged): must STILL skip scan and 40s prefill!
            (self.install / "runtime" / "xc2_rt.dll").write_bytes(b"rt-v2-completely-different-size")
            r2 = FakeRunner()
            self.assertTrue(self.setup.step_shaders_package(r2, self.repo, self.install, pathlib.Path("python"), jobs=2))
            self.assertEqual(r2.cmds, [])
            self.assertEqual(prefill_calls, [])
        finally:
            self.setup.prefill_package = orig_prefill

    def test_translator_change_runs_append_only_refresh_and_drops_preload_pack(self):
        spirv = self.install / "shader_cache" / "spirv"
        spirv.mkdir(parents=True)
        (spirv / "results.tsv").write_text("hash\tresources\n0001\tr\n", encoding="utf-8")
        (spirv / "0001.spv").write_bytes(b"\x03\x02\x23\x07")
        pack = spirv.parent / "spirv.preload.pack"
        pack.write_bytes(b"stale-pack")
        (self.install / "build" / "prefilled").write_text("5 collected\n", encoding="utf-8")

        orig_prefill = self.setup.prefill_package
        def fake_prefill(r, repo, inst, py, harness, sp, jobs=0):
            (inst / "build" / "prefilled").write_text("0 collected\n", encoding="utf-8")
            return ""
        self.setup.prefill_package = fake_prefill
        try:
            # Seed manifest at translator v1.
            r1 = FakeRunner()
            self.assertTrue(self.setup.step_shaders_package(r1, self.repo, self.install, pathlib.Path("python"), jobs=2))
            self.assertTrue(pack.is_file(), "unchanged translator must keep preload pack")

            # Debug-symbol .pdb changes must NOT trigger a full 12,132-shader re-translation!
            (self.repo / "tools" / "spirv_harness" / "spirv_harness.pdb").write_bytes(b"pdb-v2-different")
            (self.repo / "tools" / "spirv_harness" / "Ryujinx.Graphics.Shader.pdb").write_bytes(b"pdb-extra")
            r_pdb = FakeRunner()
            self.assertTrue(self.setup.step_shaders_package(r_pdb, self.repo, self.install, pathlib.Path("python"), jobs=2))
            self.assertEqual(r_pdb.cmds, [], ".pdb changes must not trigger shader re-translation")
            self.assertTrue(pack.is_file(), ".pdb changes must keep preload pack")

            # Updating Spv.Generator.dll (non-Ryujinx/non-spirv_harness prefix DLL) MUST trigger --refresh and drop pack!
            (self.repo / "tools" / "spirv_harness" / "Spv.Generator.dll").write_bytes(b"spvgen-v2")
            r2 = FakeRunner()
            self.assertTrue(self.setup.step_shaders_package(r2, self.repo, self.install, pathlib.Path("python"), jobs=2))
            self.assertEqual(len(r2.cmds), 1)
            self.assertIn("--refresh", r2.cmds[0])
            self.assertFalse(pack.exists(), "stale spirv.preload.pack must be explicitly dropped after refresh")
            self.assertTrue((spirv / "0001.spv").is_file(), "cache must remain append-only (never wiped)")
        finally:
            self.setup.prefill_package = orig_prefill

    def test_build_cache_refresh_preserves_prefill_entries_and_drops_pack(self):
        blocks = self.base / "blocks"
        cache = self.base / "shader_cache" / "spirv"
        blocks.mkdir(parents=True)
        cache.mkdir(parents=True)
        (blocks / "a.code").write_bytes(b"code-a")
        (blocks / "a.ctrl").write_bytes(b"ctrl-a")
        (blocks / "manifest.tsv").write_text(
            "code_sha1_16\tcode_file\tctrl_file\naaaa1111aaaa1111\ta.code\ta.ctrl\n",
            encoding="utf-8",
        )
        # Seed cache with archive shader (old bytes) + a runtime-generated prefill shader that is NOT in manifest.tsv.
        (cache / "aaaa1111aaaa1111.spv").write_bytes(b"OLD-SPV")
        (cache / "ffff9999ffff9999.spv").write_bytes(b"PREFILL-KEEP")
        (cache / "results.tsv").write_text(
            "hash\tresources\naaaa1111aaaa1111\told-res\nffff9999ffff9999\tprefill-res\n",
            encoding="utf-8",
        )
        pack = cache.parent / "spirv.preload.pack"
        pack.write_bytes(b"stale-pack")

        # Tiny cross-platform fake harness written as a Python script wrapped via sys.executable or executable shim.
        harness_py = self.base / "fake_harness.py"
        harness_py.write_text(
            "import os, sys\n"
            "out = sys.argv[sys.argv.index('--out') + 1]\n"
            "os.makedirs(out, exist_ok=True)\n"
            "open(os.path.join(out, 'aaaa1111aaaa1111.spv'), 'wb').write(b'NEW-SPV')\n"
            "open(os.path.join(out, 'results.tsv'), 'w', encoding='utf-8').write('hash\\tresources\\naaaa1111aaaa1111\\tnew-res\\n')\n",
            encoding="utf-8",
        )
        if sys.platform == "win32":
            harness = self.base / "fake_harness.cmd"
            harness.write_text(f'@"{sys.executable}" "{harness_py}" %*\r\n', encoding="utf-8")
        else:
            harness = self.base / "fake_harness.sh"
            harness.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{harness_py}" "$@"\n', encoding="utf-8")
            harness.chmod(harness.stat().st_mode | stat.S_IXUSR)

        proc = subprocess.run(
            [
                sys.executable,
                str(ROOT / "tools" / "shader" / "cache" / "build_cache.py"),
                "--blocks", str(blocks),
                "--cache", str(cache),
                "--harness", str(harness),
                "--work", str(self.base / "work"),
                "--no-val",
                "--refresh",
            ],
            capture_output=True,
            text=True,
        )
        self.assertEqual(proc.returncode, 0, proc.stdout + "\n" + proc.stderr)
        self.assertEqual((cache / "aaaa1111aaaa1111.spv").read_bytes(), b"NEW-SPV")
        self.assertEqual((cache / "ffff9999ffff9999.spv").read_bytes(), b"PREFILL-KEEP")
        rows = (cache / "results.tsv").read_text(encoding="utf-8")
        self.assertIn("aaaa1111aaaa1111\tnew-res", rows)
        self.assertIn("ffff9999ffff9999\tprefill-res", rows)
        self.assertNotIn("old-res", rows)
        self.assertFalse(pack.exists(), "build_cache.py --refresh must explicitly drop stale spirv.preload.pack")


if __name__ == "__main__":
    unittest.main()
