"""Per-slide JPEG gallery of the most extreme-scoring tiles.

Two rows per slide: the N highest-scoring tiles (lobular side, red in the
report) and the N lowest (ductal side, blue). Each tile is cropped from the
WSI at its CSV coordinates (level-1 space, same grid as the masks), the
score is printed under it. One JPEG per slide — for eyeballing whether the
LDA axis tracks visible morphology.

Usage:
    python make_tile_gallery.py --tiles lda_scores_v6_095_val.csv \
        --wsi-root /mnt/bioptic_tree --slide 2020_08267-04-T2 --out galleries
"""

import argparse
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from make_mask_tiff import find_wsi

TILE = 224  # tile size in level-1 pixels


def _font():
    try:
        return ImageFont.load_default(size=16)
    except TypeError:  # Pillow < 10.1
        return ImageFont.load_default()


def compose_grid(top, bot, header, n):
    """top/bot: lists of (PIL image, score), already sorted best-first."""
    pad, strip = 4, 24
    w = n * TILE + (n + 1) * pad
    header_h = strip + pad
    row_h = strip + TILE + strip
    img = Image.new("RGB", (w, header_h + 2 * (row_h + pad)), "white")
    d = ImageDraw.Draw(img)
    f = _font()
    d.text((pad, pad), header, fill="black", font=f)
    for r, (cells, name) in enumerate([
        (top, "lobular side — highest scores"),
        (bot, "ductal side — lowest scores"),
    ]):
        y0 = header_h + r * (row_h + pad)
        d.text((pad, y0), name, fill="black", font=f)
        for i, (tile, score) in enumerate(cells):
            x = pad + i * (TILE + pad)
            img.paste(tile, (x, y0 + strip))
            d.text((x, y0 + strip + TILE + 2), f"{score:+.2f}", fill="black", font=f)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiles", required=True, help="CSV with slide_name,x,y,value")
    ap.add_argument("--wsi-root", required=True)
    ap.add_argument("--slide", required=True, help="slide name, comma-separated list, or 'all'")
    ap.add_argument("--out", required=True, help="output dir (one .jpg per slide)")
    ap.add_argument("-n", type=int, default=10, help="tiles per row")
    ap.add_argument("--morphology", help="optional CSV with slide_id + morphology (adds label to header)")
    args = ap.parse_args()

    import openslide  # deferred: needs libopenslide, not present on dev machines

    label = {}
    if args.morphology:
        m = pd.read_csv(args.morphology)
        m["morphology"] = m["morphology"].astype(str).str.replace(" ", "")
        label = dict(zip(m["slide_id"], m["morphology"], strict=True))

    df = pd.read_csv(args.tiles)
    slides = df["slide_name"].unique() if args.slide == "all" else [s.strip() for s in args.slide.split(",")]
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    for slide_name in slides:
        out_path = out_dir / f"{slide_name}.jpg"
        if out_path.exists():
            continue  # resume
        wsi_path = find_wsi(slide_name, args.wsi_root)
        if wsi_path is None:
            print(f"[skip] WSI not found for {slide_name}")
            continue
        st = df[df["slide_name"] == slide_name]
        if st.empty:
            print(f"[skip] no tiles for {slide_name}")
            continue

        slide = openslide.OpenSlide(str(wsi_path))
        ds = slide.level_downsamples[1] if slide.level_count > 1 else 1.0

        def crop(t):
            loc = (round(t.x * ds), round(t.y * ds))
            return slide.read_region(loc, 1, (TILE, TILE)).convert("RGB")

        top = [(crop(t), t.value) for t in st.nlargest(args.n, "value").itertuples()]
        bot = [(crop(t), t.value) for t in st.nsmallest(args.n, "value").itertuples()]
        slide.close()

        header = (f"{slide_name} | labeled {label.get(slide_name, '?')} | "
                  f"mean {st['value'].mean():+.2f} | {len(st)} tiles")
        compose_grid(top, bot, header, args.n).save(out_path, quality=88)
        print(f"{slide_name}: mean {st['value'].mean():+.2f} -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
