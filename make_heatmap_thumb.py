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
import openslide
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

    # Read at ~2000px wide using OpenSlide
    slide = openslide.OpenSlide(str(wsi_path))
    w, h = slide.dimensions
    target_w = 2000
    # Find the level closest to target_w
    level = 0
    for i, (lw, lh) in enumerate(slide.level_dimensions):
        if lw <= target_w:
            level = i
            break
        level = i
    lw, lh = slide.level_dimensions[level]
    img = slide.read_region((0, 0), level, (lw, lh))
    slide.close()

    # Scale heatmap to match image size
    # Tile coords are in level-1 pixel space; display image is full WSI at `level`
    # Scale: display_size / full_WSI_size_at_level1
    # Full WSI at level 0 is (w, h); at level 1 it's (w/2, h/2)
    # Display is (lw, lh) = (w >> level, h >> level)
    # So scale from level-1 coords to display: lw / (w/2) = lw*2/w
    scale = lw / (w / 2)  # level-1 to display scale
    hm_w = int(max_x * scale)
    hm_h = int(max_y * scale)
    # Create full-size canvas and paste scaled heatmap
    hm_full = np.zeros((hm_h, hm_w, 3), dtype=np.uint8)
    alpha_full = np.zeros((hm_h, hm_w), dtype=np.uint8)
    hm_small_img = Image.fromarray(hm)
    hm_small_img = hm_small_img.resize((hm_w, hm_h), Image.NEAREST)
    alpha_small_img = Image.fromarray(alpha)
    alpha_small_img = alpha_small_img.resize((hm_w, hm_h), Image.NEAREST)
    hm_full = np.array(hm_small_img)
    alpha_full = np.array(alpha_small_img)
    # Now resize full canvas to display size
    hm_img = Image.fromarray(hm_full).resize(img.size, Image.NEAREST)
    alpha_img = Image.fromarray(alpha_full).resize(img.size, Image.NEAREST)

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
