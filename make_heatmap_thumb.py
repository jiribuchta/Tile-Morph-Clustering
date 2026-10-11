"""Create heatmap overlay thumbnails from tile scores CSV.

Usage:
    python make_heatmap_thumb.py \
        --tiles heatmaps_tiles.csv \
        --wsi-root /mnt/bioptic_tree \
        --out thumbs/ \
        --slides 2020_00479-01-T,2020_00397-02-T
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
from PIL import Image


def make_thumb(csv_path: str, slide_name: str, wsi_root: str, out_path: str,
              tile_size: int = 112, stride: int = 112):
    """Create a JPEG thumbnail with heatmap overlay for one slide."""
    # Load tiles for this slide
    df = pd.read_csv(csv_path)
    slide_tiles = df[df["slide_name"] == slide_name]
    if slide_tiles.empty:
        print(f"  [skip] no tiles for {slide_name}")
        return

    # Build 2D grid from tile coords
    max_x = slide_tiles["x"].max() + tile_size
    max_y = slide_tiles["y"].max() + tile_size
    grid = np.zeros((max_y // stride + 1, max_x // stride + 1), dtype=np.float32)

    for _, row in slide_tiles.iterrows():
        gy = row["y"] // stride
        gx = row["x"] // stride
        if gy < grid.shape[0] and gx < grid.shape[1]:
            grid[gy, gx] = max(grid[gy, gx], row["value"])

    # Create heatmap image (red = high score)
    hm = np.zeros((*grid.shape, 3), dtype=np.uint8)
    hm[:, :, 2] = (grid * 255).astype(np.uint8)  # red channel
    # Alpha: only show where score > 0
    alpha = (grid > 0).astype(np.uint8) * 150  # semi-transparent

    # Load WSI at a low pyramid level
    # Path structure: <root>/year/NN/NNN/slide_name/slide_name.tiff
    # e.g. 2020_00479-01-T -> 2020/00/479/2020_00479-01-T/2020_00479-01-T.tiff
    # 2020_00479-01-T -> year=2020, nn=00, nnn=479
    record = slide_name.split("-")[0]  # 2020_00479
    year = record[:4]    # 2020
    nn = record[5:7]     # 00
    nnn = record[7:10]   # 479
    slide_dir = Path(wsi_root) / year / nn / nnn / slide_name
    # Try common extensions
    wsi_path = None
    for ext in (".tiff", ".mrxs.tiff", ".ndpi", ".svs"):
        candidate = slide_dir / f"{slide_name}{ext}"
        if candidate.exists():
            wsi_path = candidate
            break
    if wsi_path is None:
        print(f"  [skip] WSI not found in {slide_dir}")
        return

    # Read lowest pyramid level with tifffile (no full image load)
    # Find a level that gives ~2000px wide
    with tifffile.TiffFile(wsi_path) as tf:
        h0, w0 = tf.pages[0].shape[0], tf.pages[0].shape[1]
        import math
        target_w = 2000
        level = max(0, int(math.log2(w0 / target_w)))
        level = min(level, len(tf.pages) - 1)
        data = tf.pages[level].array
    img = Image.fromarray(data[:, :, :3] if data.ndim == 3 else data)

    # Scale heatmap to match image size
    hm_img = Image.fromarray(hm)
    hm_img = hm_img.resize(img.size, Image.NEAREST)

    # Create alpha channel
    alpha_img = Image.fromarray(alpha)
    alpha_img = alpha_img.resize(img.size, Image.NEAREST)

    # Composite: WSI + red overlay
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    overlay.paste(hm_img, (0, 0), alpha_img)
    result = Image.new("RGB", img.size)
    result.paste(img.convert("RGB"), (0, 0))
    # Blend
    base = np.array(result, dtype=np.float32)
    ov = np.array(overlay, dtype=np.float32)
    a = ov[:, :, 3:4] / 255.0
    blended = base * (1 - a) + ov[:, :, :3] * a
    result = Image.fromarray(blended.astype(np.uint8))

    out = Path(out_path) / f"{slide_name}.jpg"
    result.save(out, quality=85)
    print(f"  {out} ({result.size[0]}x{result.size[1]})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiles", required=True, help="heatmap CSV (slide_name,x,y,value)")
    ap.add_argument("--wsi-root", required=True, help="root dir with WSI TIFFs")
    ap.add_argument("--out", default="thumbs/", help="output dir for JPEGs")
    ap.add_argument("--slides", help="comma-separated slide names (default: all in CSV)")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.slides:
        slides = [s.strip() for s in args.slides.split(",")]
    else:
        df = pd.read_csv(args.tiles)
        slides = df["slide_name"].unique().tolist()

    print(f"{len(slides)} slides -> {out_dir}")
    for s in slides:
        make_thumb(args.tiles, s, args.wsi_root, str(out_dir))


if __name__ == "__main__":
    main()
