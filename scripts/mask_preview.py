"""Render a cluster label mask (.tiff, uint8, 0=bg, 1..k=cluster) as a colored PNG.

The label mask has no color, so viewers like xOpat can't show it meaningfully.
This paints each cluster value a distinct color (0=black) and downscales to a
viewable size. Same color = tiles that matched into the same cluster.

Usage:
    uv run python scripts/mask_preview.py --masks masks --dest previews
    uv run python scripts/mask_preview.py --masks masks --dest previews --max-size 4000
"""

import argparse
from pathlib import Path

import numpy as np
import tifffile


def colorize(label: np.ndarray, n: int) -> np.ndarray:
    """label (H,W) int in 0..n -> RGB uint8; 0=black, 1..n distinct hues."""
    import colorsys

    pal = np.zeros((n + 1, 3), np.uint8)  # pal[0] = black background
    for i in range(1, n + 1):
        r, g, b = colorsys.hsv_to_rgb((i * 0.61803) % 1.0, 0.85, 1.0)
        pal[i] = (round(r * 255), round(g * 255), round(b * 255))
    return pal[label]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--masks", required=True, help="dir with <slide>.tiff label masks")
    ap.add_argument("--dest", required=True, help="dir to write <slide>.png previews")
    ap.add_argument("--max-size", type=int, default=3000, help="downscale longest edge to this")
    args = ap.parse_args()

    from PIL import Image

    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)
    files = sorted(Path(args.masks).glob("*.tiff"))
    print(f"{len(files)} masks -> {dest}")
    for f in files:
        with tifffile.TiffFile(str(f)) as t:
            label = t.asarray().astype(np.int32)
        rgb = colorize(label, int(label.max()))
        img = Image.fromarray(rgb, "RGB")
        scale = max(img.size) / args.max_size
        if scale > 1:  # NEAREST: categorical, no color bleed at cluster borders
            img = img.resize(
                (round(img.width / scale), round(img.height / scale)),
                Image.Resampling.NEAREST,
            )
        out = dest / f"{f.stem}.png"
        img.save(out)
        print(f"  {f.name} -> {out.name}  ({img.width}x{img.height})")


if __name__ == "__main__":
    main()
