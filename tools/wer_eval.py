#!/usr/bin/env python3
"""Simple free-running WER evaluator for whisperhybrid.

Manifest CSV columns:
    audio,reference,language

Example:
    python3 tools/wer_eval.py \
      --manifest fleurs-el-heldout.csv \
      --bin ./app/whisperhybrid \
      --model ./app/model/baked-q4_k.bin \
      --lang el --beam 5 --out results.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List, Tuple


def normalize_text(s: str) -> str:
    s = s.lower().strip()
    # Keep Greek/Latin letters, numbers, and whitespace. Remove most punctuation.
    s = re.sub(r"[^0-9a-zα-ωάέήίόύώϊϋΐΰς\s]+", " ", s, flags=re.IGNORECASE)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def words(s: str) -> List[str]:
    s = normalize_text(s)
    return s.split() if s else []


def edit_distance(a: List[str], b: List[str]) -> int:
    # distance from reference a to hypothesis b
    prev = list(range(len(b) + 1))
    for i, aw in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, bw in enumerate(b, 1):
            cur[j] = min(
                prev[j] + 1,       # deletion
                cur[j - 1] + 1,    # insertion
                prev[j - 1] + (aw != bw),
            )
        prev = cur
    return prev[-1]


def run_one(args: argparse.Namespace, audio: str, lang: str) -> Tuple[str, str, int]:
    cmd = [
        args.bin,
        "-json",
        "-profile", args.profile,
        "-model", args.model,
        "-lang", lang or args.lang,
        "-beam", str(args.beam),
        "-gpu", str(args.gpu),
        "-cpu", str(args.cpu),
        audio,
    ]
    if args.extra:
        cmd[1:1] = args.extra
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        return "", proc.stderr.strip() or f"exit {proc.returncode}", 0
    for line in proc.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if obj.get("error"):
            return obj.get("text", ""), obj.get("error", ""), 0
        return obj.get("text", ""), "", int(obj.get("infer_ms") or 0)
    return "", "no JSON result from binary", 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--bin", required=True)
    ap.add_argument("--model", default="auto")
    ap.add_argument("--profile", default="quality")
    ap.add_argument("--lang", default="el")
    ap.add_argument("--beam", type=int, default=5)
    ap.add_argument("--gpu", type=int, default=1)
    ap.add_argument("--cpu", type=int, default=0)
    ap.add_argument("--out", default="wer_results.csv")
    ap.add_argument("--extra", nargs="*", default=[])
    args = ap.parse_args()

    total_err = 0
    total_words = 0
    rows_out = []

    with open(args.manifest, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader, 1):
            audio = row.get("audio", "")
            ref = row.get("reference", "")
            lang = row.get("language", args.lang) or args.lang
            if not audio or not ref:
                print(f"skip row {idx}: missing audio/reference", file=sys.stderr)
                continue
            hyp, err, infer_ms = run_one(args, audio, lang)
            ref_words = words(ref)
            hyp_words = words(hyp)
            dist = edit_distance(ref_words, hyp_words)
            denom = max(1, len(ref_words))
            wer = dist / denom
            total_err += dist
            total_words += len(ref_words)
            rows_out.append({
                "audio": audio,
                "language": lang,
                "wer": f"{wer:.6f}",
                "infer_ms": infer_ms,
                "errors": dist,
                "ref_words": len(ref_words),
                "hypothesis": hyp,
                "reference": ref,
                "runtime_error": err,
            })
            print(f"{idx}: WER={wer:.3f} words={len(ref_words)} {Path(audio).name}", file=sys.stderr)

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["audio", "language", "wer", "infer_ms", "errors", "ref_words", "hypothesis", "reference", "runtime_error"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_out)

    corpus_wer = total_err / max(1, total_words)
    print(json.dumps({
        "manifest": args.manifest,
        "profile": args.profile,
        "model": args.model,
        "items": len(rows_out),
        "errors": total_err,
        "words": total_words,
        "wer": corpus_wer,
        "out": args.out,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
