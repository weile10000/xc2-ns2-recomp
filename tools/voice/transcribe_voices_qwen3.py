#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
r"""Qwen3-ASR dual-source (JP + EN) transcriber for XC2 voice manifests.

Same output schema as transcribe_voices_rocm.py, so the two can share one transcripts.jsonl and the
translation pipeline does not care which model produced a line.

Why this exists next to the Whisper one: Qwen3-ASR-1.7B is ~7x faster here (12.3 vs 1.66 clips/s on a
RX 6800 XT) and adds punctuation, but it drops non-speech vocalisations (laughter, screams) and
occasionally hallucinates. So this script marks those for a Whisper re-run instead of trusting them:
every record carries needs_audio_review, and the caller re-runs just those events.

Speed notes (measured 2026-11-28, RX 6800 XT, batch sweep on 89 clips):
  - attn_implementation="sdpa" is required; ROCm exposes flash/mem_efficient/math backends for it.
  - Input features must be cast to the model dtype by hand, or conv2d raises a dtype mismatch.
  - Sorting clips by length before chunking is worth ~2x over manifest order (less padding).
  - batch 48 length-sorted: 12.28 clips/s at 4.88 GB; batch 64 was no better.

Example:
    python tools/voice/transcribe_voices_qwen3.py --romfs <romfs> \
        --manifest work/voice_pipeline/manifest_v2_full.json \
        --out work/voice_pipeline/transcripts.jsonl \
        --model Qwen/Qwen3-ASR-1.7B-hf --batch-size 48
"""
import argparse
import json
import pathlib
import re
import sys
import time

import numpy as np
import torch
from transformers import AutoProcessor, Qwen3ASRForConditionalGeneration

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import transcribe_voices_rocm as tvr  # noqa: E402  (decode helpers are shared)

_JP_RE = re.compile(r"[\u3040-\u30ff\u4e00-\u9fff]")
_EN_RE = re.compile(r"[A-Za-z]")
_HALLUCINATION = (
    "ご視聴ありがとうございました",
    "チャンネル登録",
    "thank you for watching",
    "subtitles by",
    "amara.org",
)


def strip_prefix(text: str) -> str:
    """Qwen3-ASR returns 'language Japanese<asr_text>...'; keep only the transcript."""
    i = text.find("<asr_text>")
    return (text[i + len("<asr_text>"):] if i >= 0 else text).strip()


def needs_review(text: str, dur: float, lang: str) -> bool:
    """True when the line looks like a dropped vocalisation rather than a transcript.

    Qwen3-ASR emits an empty string for laughter and writes 'Ta!' for a scream, so an empty or
    non-native-script result on real audio is a model miss, not silence in the clip.
    """
    t = text.strip()
    if not t:
        return dur >= 0.6
    if any(h in t.lower() for h in _HALLUCINATION):
        return True
    has_script = bool(_JP_RE.search(t)) if lang == "jp" else bool(_EN_RE.search(t))
    if not has_script:
        return True
    if dur >= 2.0 and len(t) <= 2:
        return True
    return False


def load_done(path: pathlib.Path) -> set[str]:
    done: set[str] = set()
    if not path.is_file():
        return done
    with path.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            ev = rec.get("event")
            if ev:
                done.add(ev)
    return done


def build_prompt(proc) -> str:
    return proc.apply_chat_template(
        [{"role": "user", "content": [{"type": "audio", "audio": None}]}],
        add_generation_prompt=True,
        tokenize=False,
    )


@torch.inference_mode()
def transcribe(model, proc, prompt: str, arrays: list[np.ndarray], dtype, device, max_new_tokens: int) -> list[str]:
    if not arrays:
        return []
    inputs = proc(text=[prompt] * len(arrays), audio=arrays, sampling_rate=16000, return_tensors="pt")
    inputs = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in inputs.items()}
    if "input_features" in inputs:
        inputs["input_features"] = inputs["input_features"].to(dtype)
    n_in = inputs["input_ids"].shape[-1]
    out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, num_beams=1)
    return [strip_prefix(proc.tokenizer.decode(o[n_in:], skip_special_tokens=True)) for o in out]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--romfs", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True, help="Output JSONL (appended; already-present events are skipped)")
    ap.add_argument("--model", default="Qwen/Qwen3-ASR-1.7B-hf",
                    help="HF id, or a local directory holding the same files")
    ap.add_argument("--tiers", nargs="+", default=["high", "banter", "banter_orphan"])
    ap.add_argument("--packs", nargs="+", default=["main", "ira"])
    ap.add_argument("--speakers", nargs="*", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=48)
    ap.add_argument("--max-new-tokens", type=int, default=160)
    ap.add_argument("--limit", type=int, default=0, help="Stop after N events (0 = all)")
    ap.add_argument("--min-seconds", type=float, default=0.35)
    args = ap.parse_args()

    romfs = pathlib.Path(args.romfs)
    out_path = pathlib.Path(args.out)
    manifest = json.loads(pathlib.Path(args.manifest).read_text(encoding="utf-8"))["entries"]

    done = load_done(out_path)
    todo = [
        e for e in manifest
        if e["tier"] in args.tiers
        and e.get("pack") in args.packs
        and e["event"] not in done
        and (not args.speakers or e["speaker_id"] in args.speakers)
    ]
    if args.limit:
        todo = todo[: args.limit]
    if not todo:
        print("nothing to do")
        return 0
    print(f"to transcribe: {len(todo)} (skipping {len(done)} already in {out_path.name})")

    jp_pck, jp_wems = tvr.load_pck_wems(romfs / "sound" / "jp.pck")
    en_pck, en_wems = tvr.load_pck_wems(romfs / "sound" / "en.pck")

    print("loading model...")
    t0 = time.time()
    proc = AutoProcessor.from_pretrained(args.model)
    model = Qwen3ASRForConditionalGeneration.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation="sdpa", device_map={"": "cuda"}
    ).eval()
    dtype, device = next(model.parameters()).dtype, "cuda"
    prompt = build_prompt(proc)
    print(f"model ready in {time.time() - t0:.1f}s, {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B params")

    def collect(pack_pck, pack_wems, key, tag):
        got = []
        for e in todo:
            pcm = tvr.extract_pcm(pack_pck, pack_wems, e.get(key) or 0)
            if pcm is None or len(pcm) < int(args.min_seconds * 16000):
                got.append((e, None))
            else:
                got.append((e, pcm))
        return got

    print("decoding jp/en audio...")
    jp_items = collect(jp_pck, jp_wems, "jp_wem", "jp")
    en_items = collect(en_pck, en_wems, "en_wem", "en")

    texts: dict[str, dict[str, str]] = {e["event"]: {} for e in todo}
    for lang, items in (("jp", jp_items), ("en", en_items)):
        work = [(e, pcm) for e, pcm in items if pcm is not None]
        work.sort(key=lambda x: len(x[1]))  # length-sorted chunking: ~2x over manifest order
        total = len(work)
        print(f"{lang}: {total} clip(s) to transcribe")
        started = time.time()
        for i in range(0, total, args.batch_size):
            chunk = work[i : i + args.batch_size]
            outs = transcribe(model, proc, prompt, [p for _, p in chunk], dtype, device, args.max_new_tokens)
            for (e, _), txt in zip(chunk, outs):
                texts[e["event"]][lang] = txt
            done_n = min(i + args.batch_size, total)
            rate = done_n / max(time.time() - started, 1e-6)
            print(f"  {lang} {done_n}/{total}  {rate:.1f} clips/s", flush=True)

    review = 0
    with out_path.open("a", encoding="utf-8") as f:
        for e in todo:
            ev = e["event"]
            jp = texts[ev].get("jp", "")
            en = texts[ev].get("en", "")
            flag = needs_review(jp, e.get("jp_dur") or 0.0, "jp") or needs_review(en, e.get("en_dur") or 0.0, "en")
            review += int(flag)
            f.write(json.dumps({
                "event": ev,
                "pack": e.get("pack"),
                "speaker_id": e["speaker_id"],
                "kk": e["kk"],
                "tier": e["tier"],
                "jp_dur": e.get("jp_dur", 0.0),
                "en_dur": e.get("en_dur", 0.0),
                "jp_text": jp,
                "en_text": en,
                "needs_audio_review": flag,
                "model": "qwen3-asr-1.7b",
            }, ensure_ascii=False) + "\n")
    print(f"wrote {len(todo)} record(s); flagged for Whisper re-run: {review}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
