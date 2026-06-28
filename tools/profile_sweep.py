#!/usr/bin/env python3
"""Run whisperhybrid profiles and compare WER + latency.

Manifest CSV columns:
    audio,reference,language

Example:
    python3 tools/profile_sweep.py \
      --manifest fleurs-el-heldout.csv \
      --bin ./app/whisperhybrid \
      --models-dir ./app/model \
      --lang el \
      --out profile_sweep.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


def normalize_text(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"[^0-9a-zα-ωάέήίόύώϊϋΐΰς\s]+", " ", s, flags=re.IGNORECASE)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def words(s: str) -> List[str]:
    s = normalize_text(s)
    return s.split() if s else []


def edit_distance(a: List[str], b: List[str]) -> int:
    prev = list(range(len(b) + 1))
    for i, aw in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, bw in enumerate(b, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (aw != bw))
        prev = cur
    return prev[-1]


def find_model(models_dir: Path, profile: str) -> str:
    if not models_dir.exists():
        return "auto"
    files = sorted([p for p in models_dir.iterdir() if p.suffix.lower() in {".bin", ".gguf"}])
    if not files:
        return "auto"
    if profile in {"quality", "balanced", "server"}:
        patterns = ["baked" + "q4_k", "q4_k", "q5_k", "q3_k"]
    else:
        patterns = ["q3_k", "q4_k", "q5_k"]
    names = [(p, p.name.lower()) for p in files]
    for pat in patterns:
        if pat == "bakedq4_k":
            for p, name in names:
                if "baked" in name and "q4_k" in name:
                    return str(p)
            continue
        for p, name in names:
            if pat in name:
                return str(p)
    return str(files[0])


def run_one(bin_path: str, model: str, profile: str, audio: str, lang: str, gpu: int, cpu: int) -> Tuple[str, int, str]:
    cmd = [
        bin_path,
        "-json",
        "-profile", profile,
        "-model", model,
        "-lang", lang,
        "-gpu", str(gpu),
        "-cpu", str(cpu),
        audio,
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        return "", 0, proc.stderr.strip() or f"exit {proc.returncode}"
    for line in proc.stdout.splitlines():
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        return obj.get("text", ""), int(obj.get("infer_ms") or 0), obj.get("error", "") or ""
    return "", 0, "no JSON result from binary"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--bin", required=True)
    ap.add_argument("--models-dir", default="app/model")
    ap.add_argument("--profiles", nargs="+", default=["quality", "balanced", "speed", "tiny"])
    ap.add_argument("--lang", default="el")
    ap.add_argument("--gpu", type=int, default=1)
    ap.add_argument("--cpu", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="profile_sweep.csv")
    args = ap.parse_args()

    rows = []
    with open(args.manifest, newline="", encoding="utf-8") as f:
        for idx, row in enumerate(csv.DictReader(f), 1):
            if args.limit and idx > args.limit:
                break
            audio = row.get("audio", "")
            reference = row.get("reference", "")
            lang = row.get("language", args.lang) or args.lang
            if not audio or not reference:
                print(f"skip row {idx}: missing audio/reference", file=sys.stderr)
                continue
            for profile in args.profiles:
                model = find_model(Path(args.models_dir), profile)
                hyp, infer_ms, runtime_error = run_one(args.bin, model, profile, audio, lang, args.gpu, args.cpu)
                ref_words = words(reference)
                hyp_words = words(hyp)
                errors = edit_distance(ref_words, hyp_words)
                wer = errors / max(1, len(ref_words))
                model_size_mb = 0.0
                if model != "auto" and Path(model).exists():
                    model_size_mb = Path(model).stat().st_size / (1024 * 1024)
                rows.append({
                    "profile": profile,
                    "model": model,
                    "model_size_mb": f"{model_size_mb:.1f}",
                    "audio": audio,
                    "language": lang,
                    "infer_ms": infer_ms,
                    "wer": f"{wer:.6f}",
                    "errors": errors,
                    "ref_words": len(ref_words),
                    "hypothesis": hyp,
                    "reference": reference,
                    "runtime_error": runtime_error,
                })
                print(f"{idx} {profile}: WER={wer:.3f} infer_ms={infer_ms} {Path(audio).name}", file=sys.stderr)

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "profile", "model", "model_size_mb", "audio", "language", "infer_ms", "wer",
            "errors", "ref_words", "hypothesis", "reference", "runtime_error",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary: Dict[str, Dict[str, float]] = {}
    for r in rows:
        p = r["profile"]
        s = summary.setdefault(p, {"errors": 0, "words": 0, "ms": 0, "n": 0, "size": 0})
        s["errors"] += int(r["errors"])
        s["words"] += int(r["ref_words"])
        s["ms"] += int(r["infer_ms"])
        s["n"] += 1
        s["size"] = max(s["size"], float(r["model_size_mb"]))

    print(json.dumps({
        p: {
            "wer": s["errors"] / max(1, s["words"]),
            "avg_infer_ms": s["ms"] / max(1, s["n"]),
            "model_size_mb": s["size"],
            "items": int(s["n"]),
        }
        for p, s in summary.items()
    }, ensure_ascii=False, indent=2))
    print(f"wrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
