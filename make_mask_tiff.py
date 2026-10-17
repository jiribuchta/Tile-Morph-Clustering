"""Create full-resolution mask TIFFs from tile scores CSV.

Rasterizes at level 1 (the tile coords' native space) and upscales 2x with
pixel replication -- byte-identical to rasterizing at level 0 directly
(tiles are constant-color, even-aligned rects), but with 4x less RAM and
4x less rasterization work.

One CSV read per invocation, any number of slides (--slide all | name1,name2);
resumable: existing .tiffs are skipped.

Usage:
    python make_mask_tiff.py \
        --tiles lda_scores.csv \
        --wsi-root /mnt/bioptic_tree \
        --slide all \
        --out masks_v6
"""

import argparse
from pathlib import Path

import numpy as np
import openslide
import pandas as pd
import pyvips

TILE = 224  # tile size in level-1 pixels
SCALE = 4.0  # LDA score -4..+4 -> 0..255 (global; 8500 mean=-1.46, 8520 mean=+1.37)


def find_wsi(slide_name: str, wsi_root: str) -> Path | None:
    record = slide_name.split("-")[0]
    year, nn, nnn = record[:4], record[5:7], record[7:10]
    slide_dir = Path(wsi_root) / year / nn / nnn / slide_name
    for ext in (".tiff", ".mrxs.tiff", ".ndpi", ".svs"):
        candidate = slide_dir / f"{slide_name}{ext}"
        if candidate.exists():
            return candidate
    return None


def make_mask(slide_tiles: pd.DataFrame, slide_name: str, wsi_root: str, out_dir: Path):
    """Create a full-resolution (level 0) mask TIFF for one slide."""
    wsi_path = find_wsi(slide_name, wsi_root)
    if wsi_path is None:
        print(f"[skip] WSI not found for {slide_name}")
        return

    slide = openslide.OpenSlide(str(wsi_path))
    w, h = slide.dimensions  # level 0
    slide.close()

    # Rasterize at level 1 (tile coords' space), then 2x pixel replication.
    mw, mh = (w + 1) // 2, (h + 1) // 2
    mask = np.zeros((mh, mw), dtype=np.uint8)
    for row in slide_tiles.itertuples():
        dx = min(max(int(row.x), 0), max(0, mw - TILE))
        dy = min(max(int(row.y), 0), max(0, mh - TILE))
        val = np.uint8(np.clip(row.value / SCALE, -1.0, 1.0) * 127.5 + 127.5)
        region = mask[dy:dy+TILE, dx:dx+TILE]
        np.maximum(region, val, out=region)

    img = pyvips.Image.new_from_array(mask).zoom(2, 2)
    if img.width != w or img.height != h:  # odd level-0 dims
        img = img.crop(0, 0, w, h)
    out_path = out_dir / f"{slide_name}.tiff"
    img.tiffsave(str(out_path), tile=True, pyramid=True, bigtiff=True, compression="lzw")

    vals = slide_tiles["value"].values
    print(f"{slide_name}: {len(slide_tiles)} tiles, score range "
          f"[{vals.min():.2f}, {vals.max():.2f}] -> {out_path} ({w}x{h})", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiles", required=True, help="CSV/parquet with slide_name,x,y,value")
    ap.add_argument("--wsi-root", required=True)
    ap.add_argument("--slide", required=True, help="slide name, comma-separated list, or 'all'")
    ap.add_argument("--out", required=True, help="output dir (one .tiff per slide)")
    args = ap.parse_args()

    df = pd.read_csv(args.tiles) if str(args.tiles).endswith(".csv") else pd.read_parquet(args.tiles)
    slides = df["slide_name"].unique() if args.slide == "all" else [s.strip() for s in args.slide.split(",")]
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    for si, slide_name in enumerate(slides):
        if (out_dir / f"{slide_name}.tiff").exists():
            continue  # resume
        slide_tiles = df[df["slide_name"] == slide_name]
        if slide_tiles.empty:
            print(f"[skip] no tiles for {slide_name}")
            continue
        make_mask(slide_tiles, slide_name, args.wsi_root, out_dir)
        print(f"  {si+1}/{len(slides)}", flush=True)


if __name__ == "__main__":
    main()
