# SPDX-License-Identifier: GPL-3.0-or-later
"""Unit tests for raw un-trimmed NS1+NS2 dump detection and per-DLC junction normalization."""
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools" / "setup"))
import xc2_setup  # noqa: E402


class _FakeRunner:
    def __init__(self, dry=False):
        self.dry = dry
        self.logs = []

    def log(self, msg):
        self.logs.append(msg)


class TestRawDumpAndDlc(unittest.TestCase):
    def test_find_romfs_prefers_ns2_upgrade_pack_over_ns1_sibling(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "exefs").mkdir()
            (root / "romfs").mkdir()
            (root / "romfs" / "bf2.ard").write_bytes(b"ns1")
            upgrade = root / "0100e95004039062" / "0" / "updates" / "65536" / "program" / "0" / "romfs"
            upgrade.mkdir(parents=True)
            (upgrade / "bf2.ard").write_bytes(b"ns2-full-romfs")

            got = xc2_setup.find_romfs(root, None)
            self.assertEqual(got, upgrade)
            # Explicit --romfs still wins when valid.
            self.assertEqual(xc2_setup.find_romfs(root, root / "romfs"), root / "romfs")

    def test_discover_dlc_romfs_nested_and_skips_upgrade_pack(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            torna = root / "0100e95004039001" / "0" / "data" / "0" / "romfs"
            jp_voice = root / "0100E95004039063" / "0" / "updates" / "65536" / "data" / "0" / "romfs"
            upgrade = root / "0100e95004039062" / "0" / "updates" / "65536" / "program" / "0" / "romfs"
            for p in (torna, jp_voice, upgrade):
                p.mkdir(parents=True)

            found = dict(xc2_setup.discover_dlc_romfs(root))
            self.assertEqual(found, {
                "0100e95004039001": torna,
                "0100e95004039063": jp_voice,
            })

    @unittest.skipUnless(sys.platform == "win32", "NTFS junction test requires Windows")
    def test_link_dlc_normalizes_nested_layout_and_replaces_legacy_junction(self):
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            dump = base / "raw_dump"
            install = base / "install"
            torna = dump / "0100e95004039001" / "0" / "data" / "0" / "romfs"
            jp_voice = dump / "0100e95004039063" / "romfs"
            upgrade = dump / "0100e95004039062" / "0" / "updates" / "65536" / "program" / "0" / "romfs"
            for p in (torna, jp_voice, upgrade):
                p.mkdir(parents=True)
            (torna / "torna.bin").write_bytes(b"ok")
            (jp_voice / "jp.bin").write_bytes(b"ok")

            # Seed a legacy whole-folder junction at install/game/dlc-301 -> dump
            legacy = install / "game" / "dlc-301"
            legacy.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(["cmd", "/c", "mklink", "/J", str(legacy), str(dump)],
                           check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            old_env = os.environ.get("XC2_AOC_DIR")
            os.environ["XC2_AOC_DIR"] = str(dump)
            try:
                r = _FakeRunner()
                xc2_setup.link_dlc(install, r)
            finally:
                if old_env is None:
                    os.environ.pop("XC2_AOC_DIR", None)
                else:
                    os.environ["XC2_AOC_DIR"] = old_env

            self.assertTrue((legacy / "0100e95004039001" / "romfs" / "torna.bin").is_file())
            self.assertTrue((legacy / "0100e95004039063" / "romfs" / "jp.bin").is_file())
            self.assertFalse((legacy / "0100e95004039062").exists())


if __name__ == "__main__":
    unittest.main()
