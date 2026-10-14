"""Create full-resolution mask TIFF from tile scores CSV.

Usage:
    python make_mask_tiff.py \
        --tiles lda_scores.csv \
        --wsi-root /mnt/bioptic_tree \
        --slide 2020_00479-01-T \
        --out mask_2020_00479-01-T.tiff
"""

import argparse
from pathlib import Path

import numpy as np
import openslide
import pandas as pd
import pyvips


def make_mask(csv_path: str, slide_name: str, wsi_root: str, out_path: str, level: int = 1):
    """Create a mask TIFF from tile scores at the given WSI pyramid level."""
    df = pd.read_csv(csv_path) if str(csv_path).endswith(".csv") else pd.read_parquet(csv_path)
    slide_tiles = df[df["slide_name"] == slide_name]
    if slide_tiles.empty:
        print(f"[skip] no tiles for {slide_name}")
        return

    # Find WSI
    record = slide_name.split("-")[0]
    year, nn, nnn = record[:4], record[5:7], record[7:10]
    slide_dir = Path(wsi_root) / year / nn / nnn / slide_name
    wsi_path = None
    for ext in (".tiff", ".mrxs.tiff", ".ndpi", ".svs"):
        candidate = slide_dir / f"{slide_name}{ext}"
        if candidate.exists():
            wsi_path = candidate
            break
    if wsi_path is None:
        print(f"[skip] WSI not found in {slide_dir}")
        return

    # Get WSI dimensions at the requested level. Tile coords are in level-1
    # pixels, so scale = how much bigger this level's pixels are than level 1's.
    slide = openslide.OpenSlide(str(wsi_path))
    w, h = slide.level_dimensions[level]
    scale = slide.level_downsamples[level] / slide.level_downsamples[1]
    slide.close()
    print(f"WSI: {w}x{h} (level {level})")

    # Create mask at this level's resolution (uint8 to save memory)
    mask = np.zeros((h, w), dtype=np.uint8)

    # Tile size in level-1 pixels (default 224)
    tile_size = 224
    scale_x = scale_y = scale

    # Fixed global scale (v5 training stats: 8500 mean=-1.46, 8520 mean=+1.37,
    # std ~= 1.8). Score -4..+4 maps to 0..255; same pixel value = same meaning
    # on every slide. 128 (white) = genuinely ambiguous.
    SCALE = 4.0
    vals = slide_tiles["value"].values

    for i, (_, row) in enumerate(slide_tiles.iterrows()):
        dx = int(row["x"] * scale_x)
        dy = int(row["y"] * scale_y)
        dw = int(tile_size * scale_x)
        dh = int(tile_size * scale_y)
        dx = min(max(dx, 0), w - dw)
        dy = min(max(dy, 0), h - dh)
        val = np.clip(vals[i] / SCALE, -1.0, 1.0) * 127.5 + 127.5
        val = val.astype(np.uint8)
        mask[dy:dy+dh, dx:dx+dw] = np.maximum(mask[dy:dy+dh, dx:dx+dw], val)

    # Save as pyramid TIFF (openslide-compatible)
    img = pyvips.Image.new_from_array(mask)
    img.tiffsave(out_path, tile=True, pyramid=True, bigtiff=True, compression="lzw")
    print(f"Saved: {out_path} ({w}x{h})")
    print(f"  LDA score range: [{vals.min():.3f}, {vals.max():.3f}]")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiles", required=True, help="CSV with slide_name,x,y,value")
    ap.add_argument("--wsi-root", required=True)
    ap.add_argument("--slide", required=True, help="slide name")
    ap.add_argument("--out", required=True, help="output .tiff path")
    ap.add_argument("--level", type=int, default=1,
                    help="mask resolution = WSI pyramid level (1 = quarter pixels of level 0)")
    args = ap.parse_args()

    make_mask(args.tiles, args.slide, args.wsi_root, args.out, args.level)


if __name__ == "__main__":
    main()
