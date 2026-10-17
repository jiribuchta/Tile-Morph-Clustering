"""Create JPEG heatmap previews from tile scores (fast alternative to BigTIFF masks).

Overlays the LDA score per tile on a ~2000px WSI thumbnail: blue = ductal,
red = lobular, same ±4 scale and colormap as the xOpat report masks.
--band hides the ambiguous middle (|score| < ~0.6) like the report does.

Usage:
    python make_heatmap_thumb.py \
        --tiles lda_scores_realigned_val.csv \
        --wsi-root /mnt/bioptic_tree \
        --out thumbs/ \
        --slides 2020_00479-01-T,2020_00397-02-T
"""

import argparse
from pathlib import Path

import numpy as np
import openslide
import pandas as pd
from PIL import Image

# Same global scale as make_mask_tiff: score -4..+4 -> 0..255
SCORE_SCALE = 4.0
# xOpat report colormap: blue (#2166AC) -> light (#F7F7F7) -> red (#B2182B)
LUT = np.zeros((256, 3), dtype=np.uint8)
_t = np.arange(256) / 255.0
for ch, lo, hi in ((0, 0x21, 0xB2), (1, 0x66, 0x18), (2, 0xAC, 0x2B)):
    LUT[:, ch] = np.where(_t < 0.5,
                          lo + (0xF7 - lo) * _t / 0.5,
                          0xF7 + (hi - 0xF7) * (_t - 0.5) / 0.5)


def make_thumb(csv_path: str, slide_name: str, wsi_root: str, out_dir: Path,
               band: bool = False, tile_size: int = 224):
    """Create a JPEG thumbnail with LDA heatmap overlay for one slide."""
    df = pd.read_csv(csv_path) if str(csv_path).endswith(".csv") else pd.read_parquet(csv_path)
    slide_tiles = df[df["slide_name"] == slide_name]
    if slide_tiles.empty:
        print(f"  [skip] no tiles for {slide_name}")
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
        print(f"  [skip] WSI not found in {slide_dir}")
        return

    # Read a ~2000px-wide level
    slide = openslide.OpenSlide(str(wsi_path))
    w, h = slide.dimensions
    level = 0
    for i, (lw, lh) in enumerate(slide.level_dimensions):
        level = i
        if lw <= 2000:
            break
    lw, lh = slide.level_dimensions[level]
    img = slide.read_region((0, 0), level, (lw, lh)).convert("RGB")
    slide.close()

    # Tile coords are in level-1 pixels; scale to display level
    scale_x = lw / (w / 2)
    scale_y = lh / (h / 2)

    hm = np.zeros((lh, lw, 3), dtype=np.uint8)
    score = np.full((lh, lw), 128, dtype=np.uint8)  # u8 score per display pixel
    alpha = np.zeros((lh, lw), dtype=np.uint8)
    for row in slide_tiles.itertuples():
        dw = max(1, int(tile_size * scale_x))
        dh = max(1, int(tile_size * scale_y))
        dx = min(max(int(row.x * scale_x), 0), max(0, lw - dw))
        dy = min(max(int(row.y * scale_y), 0), max(0, lh - dh))
        u8 = int(np.clip(row.value / SCORE_SCALE, -1.0, 1.0) * 127.5 + 127.5)
        # overlapping tiles (stride 112): the more extreme score wins, like
        # np.maximum in the BigTIFF masks
        s = score[dy:dy+dh, dx:dx+dw]
        m = np.abs(u8 - 128) > np.abs(s.astype(np.int16) - 128)
        s[m] = u8
        hm[dy:dy+dh, dx:dx+dw][m] = LUT[u8]
        ar = alpha[dy:dy+dh, dx:dx+dw]
        ar[...] = np.maximum(ar, 150)

    if band:  # hide ambiguous middle (|score| < ~0.6), like the xOpat report
        alpha[(np.abs(score.astype(np.int16) - 128) < 18) & (alpha > 0)] = 0

    # Composite: WSI + heatmap
    base = np.array(img, dtype=np.float32)
    a = (alpha / 255.0)[:, :, None]
    blended = base * (1 - a) + hm.astype(np.float32) * a
    result = Image.fromarray(blended.astype(np.uint8))

    out = out_dir / f"{slide_name}.jpg"
    result.save(out, quality=85)
    print(f"  {out} ({result.size[0]}x{result.size[1]})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiles", required=True, help="scores CSV/parquet (slide_name,x,y,value)")
    ap.add_argument("--wsi-root", required=True, help="root dir with WSI TIFFs")
    ap.add_argument("--out", default="thumbs/", help="output dir for JPEGs")
    ap.add_argument("--slides", help="comma-separated slide names (default: all in CSV)")
    ap.add_argument("--band", action="store_true", help="hide ambiguous middle band (like xOpat)")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.slides:
        slides = [s.strip() for s in args.slides.split(",")]
    else:
        df = pd.read_csv(args.tiles) if str(args.tiles).endswith(".csv") else pd.read_parquet(args.tiles)
        slides = df["slide_name"].unique().tolist()

    print(f"{len(slides)} slides -> {out_dir}")
    for s in slides:
        make_thumb(args.tiles, s, args.wsi_root, out_dir, band=args.band)


if __name__ == "__main__":
    main()
