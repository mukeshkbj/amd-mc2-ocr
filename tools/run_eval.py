#!/usr/bin/env python3
"""Grader-style eval over the extracted MC2 samples (or any labeled set).

Runs the real entrypoint once per image, then applies the grader's
normalization (uppercase, strip whitespace and - . · _) before comparing.

Usage (inside pod/container, daemon must be reachable — app.py self-heals if
it isn't):

    python3 tools/run_eval.py \
        --images ../mc2-samples --expected ../mc2-samples/expected.json \
        --outdir /tmp/mc2-out \
        [--script /app/app.py] [--timeout 30]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

_STRIP = re.compile(r"[\s\-\.·_]")


def normalize(s: str) -> str:
    return _STRIP.sub("", s.upper())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", type=Path, required=True)
    ap.add_argument("--expected", type=Path, required=True)
    ap.add_argument("--outdir", type=Path, default=Path(tempfile.mkdtemp()))
    ap.add_argument("--script", default="/app/app.py")
    ap.add_argument("--timeout", type=float, default=30.0)
    ap.add_argument("--sock", help="daemon endpoint, e.g. tcp:127.0.0.1:9802")
    ap.add_argument("--ready", type=Path, help="daemon ready-file path")
    args = ap.parse_args()

    env = os.environ.copy()
    if args.sock:
        env["MC2_SOCK"] = args.sock
    if args.ready:
        env["MC2_READY"] = str(args.ready)
    ready_path = Path(env.get("MC2_READY", "/tmp/mc2-ocr.ready"))
    if not ready_path.exists():
        print(f"[ERROR] daemon is not ready: {ready_path} does not exist")
        return 2
    print(f"daemon ready: {ready_path}; socket: "
          f"{env.get('MC2_SOCK', '/tmp/mc2-ocr.sock')}")

    expected = json.loads(args.expected.read_text(encoding="utf-8"))
    total = scored = 0
    for name, want in sorted(expected.items()):
        img = args.images / name
        if not img.exists():
            print(f"[skip] {name} missing"); continue
        total += 1
        t0 = time.time()
        try:
            proc = subprocess.run(
                [sys.executable, args.script, "--input-image", str(img),
                 "--output-dir", str(args.outdir)],
                capture_output=True, text=True, timeout=args.timeout, env=env,
            )
        except subprocess.TimeoutExpired:
            print(f"[TIMEOUT] {name}: exceeded {args.timeout:.0f}s limit")
            continue
        dt = time.time() - t0
        out = args.outdir / f"{img.stem}_output.json"
        if proc.returncode != 0 or not out.exists():
            print(f"[fail] {name}: exec rc={proc.returncode} "
                  f"{(proc.stderr or '').strip()[:200]}")
            continue
        if dt > args.timeout:
            print(f"[SLOW] {name}: {dt:.1f}s exceeds {args.timeout:.0f}s limit")
            continue
        got = json.loads(out.read_text(encoding="utf-8")).get("text", "")
        mark = "PASS" if normalize(got) == normalize(want) else "FAIL"
        scored += mark == "PASS"
        print(f"[{mark}] {name}: got {got!r} want {want!r} ({dt:.1f}s)")

    print(f"\n{scored}/{total} correct -> {scored * 20}/{total * 20} test points "
          f"(the official challenge set has 10 images, 200 points total)")
    return 0 if scored == total else 1


if __name__ == "__main__":
    sys.exit(main())
