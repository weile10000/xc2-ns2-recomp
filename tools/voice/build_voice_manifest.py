# SPDX-License-Identifier: GPL-3.0-or-later
r"""Build an offline XC2 voice manifest from romfs-301 BDAT + Wwise PCKs (jp/en + jp_ira/en_ira).

Extracts:
  1. All 480 main-game (Vo_WinSp) + 86 Torna (Vo_WinSpIra) multi-turn post-battle banter
     conversations in exact turn order so translators never lose dialogue context.
  2. All named-speaker voice events (Main Drivers, Story Blades, Torna Org Blades,
     29 Rare Blades, 6 DLC Blades, Torna 9 Heroes) resolved against Voice_voice.bnk /
     Voice_voice_ira.bnk in both JP and EN packs, with exact audio duration and tier tags.

Output is written to a local git-ignored path (e.g. work/voice_pipeline/manifest.json).
Never commit extracted game metadata or transcripts to the public repository.
"""
import argparse
import json
import pathlib
import struct
import sys

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import extract_bdat_speakers as ebs
import extract_wwise_voices as ewv

# Authoritative 71 named VoiceIDs from common.bdat (CHR_Dr / CHR_Bl)
MAIN_SPEAKERS = (
    [1, 2, 3, 4, 5, 6, 7, 8, 10]           # Main Drivers + Azurda
    + list(range(11, 22))                  # Story Blades (11..21)
    + list(range(22, 29))                  # Torna Org Blades (22..28)
    + list(range(31, 60))                  # 29 Rare Blades (31..59)
    + list(range(501, 507))                # 6 DLC Blades (501..506)
    + [231, 232]                           # Switch 2 Edition additions (MOMO, MOMO (i))
)
IRA_SPEAKERS = list(range(701, 710))       # Torna: The Golden Country 9 Playable Heroes

GRUNT_KK = {6, 38, 50}
BANTER_KK = {44}

# Most events are named voice_<sp><kk><nn><vv>_play, but a few characters store every line under a one-letter
# variant suffix instead: Hana JS (VoiceID 15) has 1 bare event and ~266 under "a"/"b"/"c", and the Switch 2
# Edition pair 231/232 have a handful each. Scanning only the bare name made Hana JS look like she had almost no
# voice at all (measured 2026-11-28: bare=1, a=131, b=135, c=2). Keep "" first so a bare hit stays canonical.
VOICE_SUFFIXES = ("", "a", "b", "c")


def fnv_step(h: int, bs: bytes) -> int:
    for b in bs:
        h = ((h * 16777619) ^ b) & 0xFFFFFFFF
    return h


def load_bank(pck_path: pathlib.Path, bank_name: str):
    pck = pck_path.read_bytes()
    banks, wems = ewv.parse_akpk(pck)
    bid = ewv.fnv1_32(bank_name)
    if bid not in banks:
        raise SystemExit(f"Bank {bank_name} not found in {pck_path}")
    boff, bsz, _ = banks[bid]
    _, events, actions, sounds = ewv.parse_voice_bnk(pck[boff : boff + bsz], wems)
    ev_to_wem: dict[int, int] = {}
    for eid, aids in events.items():
        for aid in aids:
            tgt = actions.get(aid, 0)
            if tgt in sounds and sounds[tgt] in wems:
                ev_to_wem[eid] = sounds[tgt]
                break
    return pck, wems, ev_to_wem


def get_wem_duration(pck: bytes, wems: dict, wem_id: int) -> float:
    woff, wsz, _ = wems[wem_id]
    raw = pck[woff : woff + wsz]
    if raw[:4] != b"RIFF":
        return 0.0
    try:
        _, dur = ewv.wem_opus_to_ogg(raw)
        return round(dur, 3)
    except Exception:
        return 0.0


def fmt_voice_event(v: int) -> str:
    return f"voice_{v:08d}_Play" if v < 100_000_000 else f"voice_{v:09d}_Play"


def parse_event_meta(ev: str) -> tuple[int, int]:
    # voice_<SP><KK><NN><VV>_Play -> (speaker_id, kk)
    digits = ev[6 : ev.rfind("_Play")]
    kk = int(digits[-6:-4])
    sp = int(digits[:-6])
    return sp, kk


def extract_banter_groups(common: dict, table_name: str, nslots: int, pack_tag: str,
                          jp_ctx, en_ctx) -> tuple[list[dict], set[str]]:
    d, b = common[table_name]
    row_sz = struct.unpack_from("<H", d, b + 8)[0]
    row_off = struct.unpack_from("<H", d, b + 0x0E)[0]
    cnt = struct.unpack_from("<H", d, b + 0x10)[0]

    jp_pck, jp_wems, jp_map = jp_ctx
    en_pck, en_wems, en_map = en_ctx

    groups = []
    seen_events: set[str] = set()
    for i in range(cnt):
        p = b + row_off + i * row_sz
        chrs = struct.unpack_from(f"<{nslots}H", d, p + 4)
        vos = struct.unpack_from(f"<{nslots}I", d, p + 4 + nslots * 2)
        turns = []
        for slot_idx, (chr_id, vo_num) in enumerate(zip(chrs, vos)):
            if vo_num == 0:
                continue
            ev = fmt_voice_event(vo_num)
            sp, kk = parse_event_meta(ev)
            h = ewv.fnv1_32(ev)
            jp_wid = jp_map.get(h, 0)
            en_wid = en_map.get(h, 0)
            jp_dur = get_wem_duration(jp_pck, jp_wems, jp_wid) if jp_wid else 0.0
            en_dur = get_wem_duration(en_pck, en_wems, en_wid) if en_wid else 0.0
            turns.append({
                "turn": slot_idx + 1,
                "event": ev,
                "speaker_id": sp,
                "chr_row_id": chr_id,
                "kk": kk,
                "jp_wem": jp_wid,
                "en_wem": en_wid,
                "jp_dur": jp_dur,
                "en_dur": en_dur,
            })
            seen_events.add(ev)
        if turns:
            groups.append({
                "group_id": f"{pack_tag}_winsp_{i + 1:03d}",
                "pack": pack_tag,
                "source_table": table_name,
                "turns": turns,
            })
    return groups, seen_events


def scan_speaker_events(speakers: list[int], pack_tag: str, jp_ctx, en_ctx,
                        banter_events: set[str]) -> list[dict]:
    jp_pck, jp_wems, jp_map = jp_ctx
    en_pck, en_wems, en_map = en_ctx

    base_h = fnv_step(2166136261, b"voice_")
    play_bs = b"_play"
    d2 = [f"{i:02d}".encode() for i in range(100)]
    suf_bs = [s.encode() for s in VOICE_SUFFIXES]

    entries = []
    for sp in speakers:
        sp_s = f"{sp:02d}" if sp < 100 else f"{sp:03d}"
        h_sp = fnv_step(base_h, sp_s.encode())
        for kk in range(0, 100):
            h_kk = fnv_step(h_sp, d2[kk])
            for nn in range(0, 40):
                h_nn = fnv_step(h_kk, d2[nn])
                for vv in range(0, 100):
                    h_base = fnv_step(h_nn, d2[vv])
                    ev_bare = f"voice_{sp_s}{kk:02d}{nn:02d}{vv:02d}_Play"
                    for sf, sf_b in zip(VOICE_SUFFIXES, suf_bs):
                        h = fnv_step(fnv_step(h_base, sf_b), play_bs) if sf else fnv_step(h_base, play_bs)
                        jp_wid = jp_map.get(h, 0)
                        en_wid = en_map.get(h, 0)
                        if not jp_wid and not en_wid:
                            continue
                        ev = f"voice_{sp_s}{kk:02d}{nn:02d}{vv:02d}{sf}_Play"
                        if kk in GRUNT_KK:
                            tier = "skip_grunt"
                        elif kk in BANTER_KK:
                            # The banter table names the bare event; a suffixed take is still that turn.
                            tier = "banter" if (ev_bare in banter_events or ev in banter_events) else "banter_orphan"
                        else:
                            tier = "high"
                        jp_dur = get_wem_duration(jp_pck, jp_wems, jp_wid) if jp_wid else 0.0
                        en_dur = get_wem_duration(en_pck, en_wems, en_wid) if en_wid else 0.0
                        entries.append({
                            "event": ev,
                            "pack": pack_tag,
                            "speaker_id": sp,
                            "kk": kk,
                            "tier": tier,
                            "jp_wem": jp_wid,
                            "en_wem": en_wid,
                            "jp_dur": jp_dur,
                            "en_dur": en_dur,
                        })
    return entries


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--romfs", required=True, help="Path to romfs-301")
    ap.add_argument("--out", required=True, help="Output manifest JSON path (under work/)")
    ap.add_argument("--include-common-blades", action="store_true",
                    help="Also scan common blades (61..80)")
    args = ap.parse_args()

    romfs = pathlib.Path(args.romfs)
    snd = romfs / "sound"

    print("Loading common.bdat...")
    idx = ebs.load_arh_subset(romfs, {"/bdat/common.bdat"})
    common = ebs.parse_bdat_container(ebs.read_ard_file(romfs / "bf2.ard", idx["/bdat/common.bdat"]))

    print("Loading Wwise PCKs (jp, en, jp_ira, en_ira)...")
    jp_main = load_bank(snd / "jp.pck", "Voice_voice")
    en_main = load_bank(snd / "en.pck", "Voice_voice")
    jp_ira = load_bank(snd / "jp_ira.pck", "Voice_voice_ira")
    en_ira = load_bank(snd / "en_ira.pck", "Voice_voice_ira")

    print("Extracting multi-turn banter conversations (Vo_WinSp & Vo_WinSpIra)...")
    main_banter, main_banter_evs = extract_banter_groups(common, "Vo_WinSp", 3, "main", jp_main, en_main)
    ira_banter, ira_banter_evs = extract_banter_groups(common, "Vo_WinSpIra", 5, "ira", jp_ira, en_ira)

    main_sp = list(MAIN_SPEAKERS) + (list(range(61, 81)) if args.include_common_blades else [])
    print(f"Scanning named-speaker events (main={len(main_sp)}, ira={len(IRA_SPEAKERS)})...")
    main_entries = scan_speaker_events(main_sp, "main", jp_main, en_main, main_banter_evs)
    ira_entries = scan_speaker_events(IRA_SPEAKERS, "ira", jp_ira, en_ira, ira_banter_evs)

    # Ensure any Vo_WinSp events outside scanned ranges are also in entries
    have_ev = {e["event"] for e in main_entries} | {e["event"] for e in ira_entries}
    extra_banter = 0
    for grp in main_banter + ira_banter:
        for t in grp["turns"]:
            if t["event"] not in have_ev:
                have_ev.add(t["event"])
                extra_banter += 1
                target = main_entries if grp["pack"] == "main" else ira_entries
                target.append({
                    "event": t["event"],
                    "pack": grp["pack"],
                    "speaker_id": t["speaker_id"],
                    "kk": t["kk"],
                    "tier": "banter",
                    "jp_wem": t["jp_wem"],
                    "en_wem": t["en_wem"],
                    "jp_dur": t["jp_dur"],
                    "en_dur": t["en_dur"],
                })

    all_entries = main_entries + ira_entries
    tier_counts: dict[str, int] = {}
    for e in all_entries:
        tier_counts[e["tier"]] = tier_counts.get(e["tier"], 0) + 1

    manifest = {
        "summary": {
            "total_entries": len(all_entries),
            "main_entries": len(main_entries),
            "ira_entries": len(ira_entries),
            "banter_groups_main": len(main_banter),
            "banter_groups_ira": len(ira_banter),
            "extra_banter_events": extra_banter,
            "by_tier": tier_counts,
        },
        "banter_groups": main_banter + ira_banter,
        "entries": all_entries,
    }

    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {out_path}")
    print("Summary:", json.dumps(manifest["summary"], indent=2))


if __name__ == "__main__":
    main()
