#!/usr/bin/env python3
"""Generate degraded copies of the sample set — the graded set is 'deliberately
harder', so test against noise/blur/glare/low-light/downscale/off-axis variants.

    python3 tools/make_harder.py --src ../mc2-samples --dst ../mc2-samples-hard
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from PIL import Image, ImageEnhance, ImageFilter


def variants(img: Image.Image, rng: random.Random):
    w, h = img.size
    blur = img.filter(ImageFilter.GaussianBlur(rng.uniform(1.0, 2.5)))
    yield "blur", blur
    noisy = img.point(lambda p: p)  # copy
    px = noisy.load()
    for _ in range(w * h // 8):
        x, y = rng.randrange(w), rng.randrange(h)
        px[x, y] = tuple(rng.randrange(256) for _ in range(3))
    yield "noise", noisy
    dark = ImageEnhance.Brightness(img).enhance(0.35)
    yield "lowlight", dark
    glare = ImageEnhance.Brightness(img).enhance(1.6)
    yield "glare", glare
    small = img.resize((max(8, w // 3), max(8, h // 3)))
    yield "lowres", small.resize((w, h))
    ang = rng.uniform(-12, 12)
    yield "angle", img.rotate(ang, expand=True, fillcolor=(80, 80, 80))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--dst", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    args.dst.mkdir(parents=True, exist_ok=True)
    expected = json.loads((args.src / "expected.json").read_text(encoding="utf-8"))
    manifest = {}
    for name, want in expected.items():
        img = Image.open(args.src / name).convert("RGB")
        for tag, v in variants(img, rng):
            out_name = f"{Path(name).stem}_{tag}.png"
            v.save(args.dst / out_name)
            manifest[out_name] = want
    (args.dst / "expected.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"wrote {len(manifest)} degraded images + expected.json to {args.dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
