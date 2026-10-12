"""Show mask TIFFs as small JPEG previews.

Usage:
    python show_masks.py --masks-dir /path/to/masks [--out /path/to/previews]
"""

import argparse
from pathlib import Path
import openslide
from PIL import Image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--masks-dir", required=True)
    ap.add_argument("--out", default=None, help="output dir for previews (default: <masks-dir>/previews)")
    ap.add_argument("--size", type=int, default=500, help="preview width in px")
    args = ap.parse_args()

    masks_dir = Path(args.masks_dir)
    out_dir = Path(args.out) if args.out else masks_dir / "previews"
    out_dir.mkdir(parents=True, exist_ok=True)

    for tiff in sorted(masks_dir.glob("*.tiff")):
        slide = openslide.OpenSlide(str(tiff))
        # Find the level closest to args.size width
        levels = slide.mpyramid_levels
        best = min(levels, key=lambda l: abs(l[0] - args.size))
        img = slide.read_region((0, 0), best[2], (best[0], best[1]))
        img = img.convert("L")
        out_path = out_dir / f"{tiff.stem}.jpg"
        img.save(out_path, quality=85)
        print(f"{out_path} ({img.width}x{img.height})")
        slide.close()

    print(f"\n{len(list(out_dir.glob('*.jpg')))} previews in {out_dir}")


if __name__ == "__main__":
    main()
