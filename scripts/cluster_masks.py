"""Per-slide cluster masks (.tiff, xOpat-ready) from a clustering run.

Reads <out>/assignments.parquet (slide_id, x, y, cluster) + <out>/slides.parquet
(slide_id, path, level, tile_extent_x, tile_extent_y, mpp_x), paints each slide's
tiles into a uint8 canvas at the tiling level (value = cluster+1, 0 = background),
and saves with ratiopath.masks.write_big_tiff — the same pyramid-BigTIFF writer
ratiopath uses for xOpat overlays (512x512 tiles, DEFLATE, xres/yres from mpp).
Equal pixel values = tiles that matched into the same cluster.

Usage:
    uv run python scripts/cluster_masks.py --out clustering/k32 --dest masks
    uv run python scripts/cluster_masks.py --out clustering/k32 --dest masks --limit 5
    uv run python scripts/cluster_masks.py --out clustering/k32 --dest masks --slides <slide_id>
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


def wsi_dims_at(path, mpp_x: float):
    """(W, H) at the WSI level whose mpp best matches mpp_x; None if unreadable."""
    try:
        import openslide

        with openslide.OpenSlide(str(path)) as s:
            base_mpp = s.mpp[0]
            best, best_diff = None, None
            for i, (w, h) in enumerate(s.level_dimensions):
                m = base_mpp / (2**i)
                diff = abs(m - mpp_x) / mpp_x
                if best_diff is None or diff < best_diff:
                    best, best_diff = (w, h), diff
            return best
    except Exception:
        return None


def build_canvas(df_slide: pd.DataFrame, slide_row: pd.Series):
    """uint8 canvas (value=cluster+1) at the tiling level; (canvas, W, H, source)."""
    tw, th = int(slide_row["tile_extent_x"]), int(slide_row["tile_extent_y"])
    dtype = np.uint16 if int(df_slide["cluster"].max()) >= 255 else np.uint8
    x = df_slide["x"].to_numpy()
    y = df_slide["y"].to_numpy()
    c = df_slide["cluster"].to_numpy()

    W, H = int(x.max() + tw), int(y.max() + th)  # fallback: tile bounds
    source = "bounds"
    dims = wsi_dims_at(slide_row["path"], float(slide_row["mpp_x"]))
    if dims is not None:  # exact WSI canvas -> perfect xOpat alignment
        W, H, source = max(W, dims[0]), max(H, dims[1]), "wsi"

    canvas = np.zeros((H, W), dtype)
    x1 = np.clip(x, 0, W)
    y1 = np.clip(y, 0, H)
    x2 = np.minimum(x + tw, W)
    y2 = np.minimum(y + th, H)
    for xi, xj, yi, yj, v in zip(x1, x2, y1, y2, c, strict=True):
        if xj > xi and yj > yi:
            canvas[yi:yj, xi:xj] = int(v) + 1
    return canvas, W, H, source


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="clustering out dir (assignments+slides parquets)")
    ap.add_argument("--dest", required=True, help="dir to write <slide_id>.tiff into")
    ap.add_argument("--limit", type=int, default=0, help="only first N slides (0 = all)")
    ap.add_argument("--slides", nargs="*", help="only these slide_ids")
    ap.add_argument("--rgb", action="store_true",
                    help="also write <name>.rgb.tiff (RGB colored, viewable in xOpat)")
    args = ap.parse_args()

    import pyvips
    from ratiopath.masks.write_big_tiff import write_big_tiff

    from mask_preview import colorize  # same dir as this script; one palette everywhere

    a = pd.read_parquet(f"{args.out}/assignments.parquet")
    s = pd.read_parquet(f"{args.out}/slides.parquet")
    s = s.set_index("slide_id")
    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)

    slides = a["slide_id"].unique()
    if args.slides:
        slides = np.intersect1d(slides, args.slides)
    if args.limit:
        slides = slides[: args.limit]
    print(f"{len(slides)} slides to write -> {dest}")

    t0 = time.monotonic()
    used_names = set()
    failed = []
    for i, sid in enumerate(slides, 1):
        row = s.loc[sid]
        df = a[a["slide_id"] == sid]
        try:
            canvas, W, H, source = build_canvas(df, row)
            stem = Path(row["path"]).stem  # same name as the original WSI
            name, n = stem, 1
            while name in used_names:
                name = f"{stem}-{n}"
                n += 1
            used_names.add(name)
            path = dest / f"{name}.tiff"
            img = pyvips.Image.new_from_array(canvas)
            # canvas is the tiling level itself -> its mpp is the slide's mpp
            mpp_y = float(row["mpp_y"]) if "mpp_y" in row else float(row["mpp_x"])
            write_big_tiff(img, path, float(row["mpp_x"]), mpp_y)
            if args.rgb:
                rgb = colorize(canvas, int(canvas.max()))
                rgb_img = pyvips.Image.new_from_array(rgb)
                write_big_tiff(rgb_img, dest / f"{name}.rgb.tiff", float(row["mpp_x"]), mpp_y)
                del rgb, rgb_img
        except Exception as e:  # one bad WSI must not kill the batch
            failed.append((sid, str(e)))
            print(f"  [{i}/{len(slides)}] {sid[:12]}.. SKIPPED: {e}")
            continue
        del canvas, img
        print(f"  [{i}/{len(slides)}] {sid[:12]}.. {W}x{H} {source} "
              f"{len(df)} tiles {path.stat().st_size / 1e6:.0f} MB "
              f"({time.monotonic() - t0:.0f}s)")
    print(f"done: {len(slides) - len(failed)} written in {time.monotonic() - t0:.0f}s")
    if failed:
        print(f"WARNING: {len(failed)} slides failed:")
        for sid, err in failed[:10]:
            print(f"  {sid[:12]}.. {err}")
        sys.exit(1)


if __name__ == "__main__":
    main()
