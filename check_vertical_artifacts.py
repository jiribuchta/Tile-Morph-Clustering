"""Diagnose vertical stripe artifacts in LDA score masks.

For each slide, two questions:

1. Was the scores CSV misaligned? (the emb[:n_keep] bug: scores of the first
   N tiles of the whole slide written at carcinoma-tile coords). Recomputes
   correctly joined scores from .pt + direction; Spearman rho ~ 1 vs the CSV
   means the CSV was fine, low rho = it was misaligned.
2. Do the CORRECTED scores still have vertical structure (scanner stripes)?
   Permutation test on per-column (x) mean variance, per-row (y) as control.

Writes corrected per-tile scores to --out (same schema as lda_scores CSV,
ready for make_mask_tiff.py).

Usage:
    python check_vertical_artifacts.py \
        --pt-dir "/mnt/projects/breast_cancer/bc/tile_embeddings/virchow2/MMCI B20-24 Val" \
        --heatmaps "/mnt/projects/breast_cancer/bc/heatmaps/virchow2_heatmaps_val" \
        --scores /mnt/projects/breast_cancer/tile_morph_clustering/lda_scores_v5_095_val.csv \
        --direction /mnt/projects/breast_cancer/tile_morph_clustering/lda_direction_v5_095_train.npz \
        --slides 2020_08267-04-T2 2023_08660-2-T \
        --out lda_scores_realigned_val.csv
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from lda_projection import load_slide, select_tiles

TILE = 224


def perm_var_test(keys, vals, n_perm=200, seed=0):
    """Variance of group means + permutation p-value (shuffle vals over tiles)."""
    rng = np.random.default_rng(seed)
    k = keys - keys.min()
    n_groups = int(k.max()) + 1

    def stat(v):
        s = np.bincount(k, weights=v, minlength=n_groups)
        c = np.bincount(k, minlength=n_groups)
        return (s[c > 0] / c[c > 0]).var()

    obs = stat(vals)
    n_ge = 1
    for _ in range(n_perm):
        if stat(vals[rng.permutation(len(vals))]) >= obs:
            n_ge += 1
    return obs, n_ge / (n_perm + 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pt-dir", required=True)
    ap.add_argument("--heatmaps", required=True, help="dir of per-slide carcinoma parquets")
    ap.add_argument("--scores", required=True, help="existing lda_scores CSV (old, possibly misaligned)")
    ap.add_argument("--direction", required=True, help="LDA direction .npz")
    ap.add_argument("--slides", nargs="*", default=None, help="subset of slide names (default: all in CSV)")
    ap.add_argument("--limit", type=int, default=15, help="max slides if --slides not given")
    ap.add_argument("--out", default="lda_scores_realigned.csv")
    args = ap.parse_args()

    d = np.load(args.direction)
    direction, scale, offset = d["direction"], float(d["scale"]), float(d["offset"])

    scores_df = pd.read_csv(args.scores)
    slides = args.slides or list(scores_df["slide_name"].unique()[: args.limit])
    pt_dir, heat_dir = Path(args.pt_dir), Path(args.heatmaps)

    corrected = []
    for si, slide_name in enumerate(slides):
        pt, parq = pt_dir / f"{slide_name}.pt", heat_dir / f"{slide_name}.parquet"
        old = scores_df[scores_df["slide_name"] == slide_name]
        if not pt.exists() or old.empty:
            print(f"[skip] {slide_name}: no .pt or no CSV rows")
            continue
        heat = pd.read_parquet(parq) if parq.exists() else None

        emb, xs, ys = load_slide(pt)
        print(f"\n{slide_name}: {emb.shape[0]} tiles in .pt, "
              f"{len(heat) if heat is not None else '?'} carcinoma(>=0.5) tiles, {len(old)} in scores CSV")
        if xs is None:
            print("  [!] no coords in .pt -- cannot realign, check manually")
            continue
        if len(old) < emb.shape[0]:
            print(f"  misaligned by construction: {len(old)} of {emb.shape[0]} tiles kept -> "
                  f"old values came from the FIRST {len(old)} tiles of the whole slide, "
                  f"painted at carcinoma-tile coords")

        emb_sel, tiles = select_tiles(emb, xs, ys, old)
        norms = np.linalg.norm(emb_sel, axis=1, keepdims=True)
        norms[norms == 0] = 1
        new = (emb_sel / norms) @ direction * scale + offset

        rho = stats.spearmanr(old["value"].values, new).statistic
        print(f"  old vs realigned: Spearman rho={rho:.3f}, "
              f"mean|diff|={np.abs(old['value'].values - new).mean():.3f} "
              f"({'MISALIGNED' if rho < 0.9 else 'was aligned'})")

        # vertical structure in corrected scores (x columns vs y rows)
        x, y = tiles["x"].values, tiles["y"].values
        cv, cp = perm_var_test(x // TILE, new)
        rv, rp = perm_var_test(y // TILE, new)
        flag = "  <- VERTICAL structure" if cp < 0.05 and cp < rp else ""
        print(f"  realigned scores: col-mean var={cv:.4f} (p={cp:.3f}), "
              f"row-mean var={rv:.4f} (p={rp:.3f}){flag}")

        # carcinoma detector: stripes should show here too if scanner-caused
        if heat is not None and len(heat) > 1:
            cv, cp = perm_var_test(heat["x"].values // TILE, heat["value"].values)
            rv, rp = perm_var_test(heat["y"].values // TILE, heat["value"].values)
            print(f"  carcinoma prob:   col-mean var={cv:.6f} (p={cp:.3f}), "
                  f"row-mean var={rv:.6f} (p={rp:.3f})"
                  + ("  <- detector sees stripes" if cp < 0.05 and cp < rp else ""))

        # rendering-gap check for make_mask_tiff (uniform 224 stride = no gaps)
        dx = np.diff(np.unique(x))
        if len(dx):
            print(f"  tile x stride: min={dx.min()}, max={dx.max()}"
                  + ("" if dx.min() == dx.max() else "  <- non-uniform, check mask gaps"))

        corrected.append(pd.DataFrame({
            "slide_name": slide_name, "x": x, "y": y, "value": new}))

    if corrected:
        out = pd.concat(corrected, ignore_index=True)
        out.to_csv(args.out, index=False)
        print(f"\nWrote {len(out)} realigned tile scores -> {args.out} (usable by make_mask_tiff.py)")


if __name__ == "__main__":
    main()
