"""Rasterize mlflow carcinoma heatmaps to per-tile values -> CSV.

Downloads the heatmaps + Val slides.parquet from mlflow, computes the mean
heatmap value per tile (level-1 grid: tile 224x224, stride 112), and writes
CSV: slide_id, slide_name, x, y, value.

Usage:
    python heatmap_to_csv.py [--out heatmaps_tiles.csv] [--min-value 0.5]
        --min-value 0.5 -> only carcinoma tiles (much smaller file)
"""

import argparse
import time
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import tifffile

TRACKING_URI = "http://mlflow.rationai-mlflow:5000"
HEATMAP_RUN = "25f15b4a379446c085c4568f2b08f703"  # Virchow2 Tile Threshold Estimation MMCI B20-24 Val
SLIDES_RUN = "569f66d87bd849129a7a0604c889ee91"  # Tile Embeddings (V2) MMCI B20-24 Val


def tile_means(a, tx, ty, sx, sy):
    """Mean of each tx x ty window at stride (sx, sy). a: uint8 (rows=y, cols=x).

    Chunked integral image over pixel rows: O(H*W) total,
    peak ~ uint8 image + 2*(chunk+ty)*W*4 bytes (float32 cumsums).
    Returns (n_y, n_x) float32 array of window means.
    """
    H, W = a.shape
    n_y = (H - ty) // sy + 1
    n_x = (W - tx) // sx + 1
    out = np.empty((n_y, n_x), dtype=np.float32)
    pad = ty - 1
    chunk = 1024  # pixel rows; peak ~5.8GB image + ~1.1GB temps
    cols = np.arange(n_x) * sx
    for r0 in range(0, H - ty + 1, chunk):  # tile tops in [r0, r1)
        r1 = min(r0 + chunk, H - ty + 1)
        i0 = -(-r0 // sy)                    # first tile index with top >= r0
        i1 = (r1 - 1) // sy                  # last tile index with top < r1
        s0 = max(0, r0 - pad)
        strip = a[s0:r1 + pad]
        # float32 cumsum is exact here: max run sum (chunk+ty)*255 < 2**24
        csum = np.cumsum(strip, axis=0, dtype=np.float32)  # (r1-s0+pad, W)
        bot = csum[ty - 1:].copy()
        bot[1:] -= csum[:-ty]
        # bot[r] = sum of strip rows r .. r+ty-1, per column
        tops = np.arange(i0, i1 + 1) * sy    # pixel rows of tile tops in this chunk
        rows = tops - s0                     # strip row index of each tile top
        col_sums = bot[rows]                  # (n, W) per-tile-row column sums
        csumx = np.cumsum(col_sums, axis=1, dtype=np.float32)
        right = csumx[:, tx - 1:].copy()
        right[:, 1:] -= csumx[:, :-tx]
        out[i0:i1 + 1] = right[:, cols] / (tx * ty)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="heatmaps_tiles.csv")
    ap.add_argument("--min-value", type=float, default=0.5,
                    help="only write tiles with value >= this (0.5 = carcinoma tiles)")
    args = ap.parse_args()

    mlflow.set_tracking_uri(TRACKING_URI)
    slides = pd.read_parquet(
        mlflow.artifacts.download_artifacts(
            f"mlflow-artifacts:/61/{SLIDES_RUN}/artifacts/MMCI B20-24 Val/slides/slides.parquet"
        )
    )
    slides["slide_id"] = slides["id"].apply(
        lambda b: b.hex() if isinstance(b, (bytes, bytearray)) else str(b)
    )
    slides["slide_name"] = slides["path"].apply(lambda p: Path(p).stem)
    print(f"{len(slides)} slides")

    heatmaps = mlflow.artifacts.download_artifacts(
        f"mlflow-artifacts:/61/{HEATMAP_RUN}/artifacts/heatmaps"
    )
    hm_files = {Path(f).stem: f for f in Path(heatmaps).glob("*.tiff")}
    print(f"{len(hm_files)} heatmaps downloaded -> {heatmaps}")

    # stream per-slide to CSV: full grid is ~218M rows, never hold it all in RAM
    n_rows = 0
    v_min, v_max, v_sum = float("inf"), float("-inf"), 0.0
    t0 = time.monotonic()
    with open(args.out, "w", newline="") as f:
        f.write("slide_id,slide_name,x,y,value\n")
        for i, s in tqdm(slides.iterrows()):
            name = s["slide_name"]
            if name not in hm_files:
                print(f"  [skip] no heatmap for {name}")
                continue
            # tifffile decodes straight into numpy: no second 5.8GB PIL buffer
            a = tifffile.imread(hm_files[name])  # uint8, rows=y, cols=x, level 1
            if a.shape != (s["extent_y"], s["extent_x"]):
                print(f"  [skip] {name}: heatmap {a.shape[::-1]} != extent {(s['extent_x'], s['extent_y'])}")
                continue
            tx, ty, sx, sy = (int(s[k]) for k in ("tile_extent_x", "tile_extent_y", "stride_x", "stride_y"))
            ox, oy = sx, sy  # original level-1 stride, for CSV coords
            # 2x2 block average first: cumsum then runs on 1/4 the pixels (~6x faster).
            # tile means preserved up to a 0.5/255 floor error (tx, sy, extents all even)
            if a.shape[0] % 2 == 0 and a.shape[1] % 2 == 0:
                b = a[0::2, 0::2].astype(np.uint16)
                b += a[0::2, 1::2]
                b += a[1::2, 0::2]
                b += a[1::2, 1::2]
                a = (b >> 2).astype(np.uint8)
                del b
                tx, ty, sx, sy = tx // 2, ty // 2, sx // 2, sy // 2
            vals = tile_means(a, tx, ty, sx, sy)  # (n_y, n_x)
            del a
            xs = np.arange(vals.shape[1]) * ox
            ys = np.arange(vals.shape[0]) * oy
            X, Y = np.meshgrid(xs, ys, indexing="xy")
            m = vals >= args.min_value
            n = int(m.sum())
            if n:
                pd.DataFrame({
                    "slide_id": s["slide_id"],
                    "slide_name": name,
                    "x": X[m].astype(np.int32),
                    "y": Y[m].astype(np.int32),
                    "value": vals[m].astype(np.float32),
                }).to_csv(f, index=False, header=False)
                n_rows += n
                v_min = min(v_min, float(vals[m].min()))
                v_max = max(v_max, float(vals[m].max()))
                v_sum += float(vals[m].sum())
            if i % 25 == 0:
                print(f"  {i+1}/{len(slides)} ({time.monotonic()-t0:.0f}s)")

    print(f"wrote {n_rows} rows -> {args.out} ({time.monotonic()-t0:.0f}s)")
    if n_rows:
        print(f"value range: {v_min:.3f}-{v_max:.3f}, mean {v_sum/n_rows:.3f}")


if __name__ == "__main__":
    main()
