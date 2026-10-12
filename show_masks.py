"""Show mask TIFFs as small JPEG previews.

Usage:
    python show_masks.py --masks-dir /path/to/masks [--out /path/to/previews]
"""

import argparse
from pathlib import Path
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
        img = Image.open(tiff)
        img = img.resize((args.size, int(img.height * args.size / img.width)), Image.LANCZOS)
        out_path = out_dir / f"{tiff.stem}.jpg"
        img.save(out_path, quality=85)
        print(f"{out_path} ({img.width}x{img.height})")

    print(f"\n{len(list(out_dir.glob('*.jpg')))} previews in {out_dir}")


if __name__ == "__main__":
    main()
