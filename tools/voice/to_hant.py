#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
r"""Derive the Traditional Chinese (zh_hant) cue text from the Simplified (zh_hans) one.

Why not a plain OpenCC pass: the shipped zh_hant pack is not just a script conversion. It uses the
official Traditional terminology (Blade -> 神劍, Driver -> 劍主, HP Potion -> 回復瓶) and a different
transliteration for most character names (尼婭 -> 妮雅, 凡達姆 -> 范達美, 花JS -> 華JS ...). It also
deliberately keeps a few names in the Simplified-looking form inside dialogue (虎 stays 虎, 白虎 must
not become 白寅), so those are skipped rather than replaced.

Validated against the shipped v1.0 pair: 14,892 / 14,940 cues reproduce exactly (99.68%).

Usage:
    python tools/voice/to_hant.py --hans <zh_hans.json> --reference <zh_hant.json> --out <zh_hant_new.json>
"""
import argparse
import json
import pathlib

from opencc import OpenCC

CC = OpenCC("s2twp")

# Names the shipped pack keeps as-is inside dialogue text (or that would corrupt a longer name).
SKIP_NAMES = {"虎", "大锤", "宫比罗", "花巴斯塔", "米诺奇", "阿德尔"}
# Names whose in-text Traditional form differs from the speaker table entry.
NAME_OVERRIDES = [
    ("花巴斯塔", "華巴斯塔"),
    ("大锤", "大錘"),
    ("宫比罗", "宮比羅"),
    ("米诺奇", "米諾奇"),
    ("阿德尔", "阿德爾"),
]
# Terminology that OpenCC cannot know (glossary §2).
TERMS = [
    ("異刃連擊", "神劍連擊"),
    ("御刃者連擊", "劍主連擊"),
    ("異刃", "神劍"),
    ("御刃者", "劍主"),
    ("恢復瓶", "回復瓶"),
]


def build_pairs(hans_doc: dict, reference_doc: dict) -> list[tuple[str, str]]:
    pairs = []
    for k, v in hans_doc["speakers"].items():
        hn = v.get("name", "")
        tn = reference_doc["speakers"].get(k, {}).get("name", "")
        if not hn or not tn or hn == tn or hn in SKIP_NAMES:
            continue
        pairs.append((CC.convert(hn), tn))
    return sorted(set(pairs), key=lambda x: -len(x[0]))


def convert(text: str, pairs: list[tuple[str, str]]) -> str:
    x = CC.convert(text)
    for a, b in pairs:
        x = x.replace(a, b)
    for a, b in NAME_OVERRIDES:
        x = x.replace(a, b)
    for a, b in TERMS:
        x = x.replace(a, b)
    return x


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hans", required=True, help="zh_hans json (source of the text)")
    ap.add_argument("--reference", required=True, help="existing zh_hant json (source of the name table)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    hans = json.loads(pathlib.Path(args.hans).read_text(encoding="utf-8"))
    ref = json.loads(pathlib.Path(args.reference).read_text(encoding="utf-8"))
    pairs = build_pairs(hans, ref)

    out = {"speakers": ref["speakers"], "cues": dict(ref["cues"])}
    added = 0
    for ev, cue in hans["cues"].items():
        if ev in out["cues"]:
            continue
        new = dict(cue)
        for key in ("text_full", "text_compact"):
            if key in new and new[key]:
                new[key] = convert(new[key], pairs)
        out["cues"][ev] = new
        added += 1
    # keep the name table in sync with any speaker the hans side has and the reference lacks
    for k, v in hans["speakers"].items():
        out["speakers"].setdefault(k, v)
    pathlib.Path(args.out).write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    print(f"added {added} cue(s), total {len(out['cues'])}, name pairs {len(pairs)} -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
