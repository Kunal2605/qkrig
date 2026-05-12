#!/usr/bin/env python3
"""
build_daily_gif.py

Combine a sorted list of PNG frames into a single animated GIF. Used by the
Docker entrypoint to roll up the 24 per-hour kriging composite PNGs into one
looping daily GIF after a full-day run.

Usage:
    python build_daily_gif.py \
        --pattern '/qkrig/exports/plots/kriging/kriging_combo_2024-09-26_*.png' \
        --output  /qkrig/exports/plots/kriging/kriging_combo_2024-09-26.gif \
        --duration 333
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

from PIL import Image


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Roll up sorted PNGs into one GIF.")
    p.add_argument("--pattern", required=True,
                   help="Glob pattern for input PNGs (lexically sorted before assembling).")
    p.add_argument("--output", required=True,
                   help="Path to the output .gif file.")
    p.add_argument("--duration", type=int, default=333,
                   help="Per-frame display duration in ms (default 333 ≈ 3 fps).")
    p.add_argument("--loop", type=int, default=0,
                   help="Loop count (0 = loop forever, 1 = play once).")
    p.add_argument("--max-width", type=int, default=None,
                   help="Optional width cap; preserves aspect ratio. Cuts file size.")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    paths = sorted(glob.glob(args.pattern))
    if not paths:
        print(f"No PNGs match {args.pattern}; nothing to render.")
        return 0
    print(f"Building GIF from {len(paths)} frame(s) → {args.output}")

    frames = []
    for p in paths:
        img = Image.open(p)
        if args.max_width and img.width > args.max_width:
            new_h = int(img.height * args.max_width / img.width)
            img = img.resize((args.max_width, new_h), Image.LANCZOS)
        # Quantize to a 256-color palette so GIF can encode it. ADAPTIVE
        # keeps each frame's palette tuned to its content; this trades a bit
        # of file size for visibly cleaner gradients.
        frames.append(img.convert("P", palette=Image.ADAPTIVE, colors=256))

    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    frames[0].save(
        args.output,
        save_all=True,
        append_images=frames[1:],
        duration=args.duration,
        loop=args.loop,
        optimize=True,
        disposal=2,
    )
    size_mb = os.path.getsize(args.output) / 1024 / 1024
    print(f"Done. {args.output}  ({size_mb:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
