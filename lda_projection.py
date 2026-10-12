"""Supervised projection: find the direction that best separates 8500 vs 8520.

Usage:
    python lda_projection.py \
        --pt-dir "/mnt/.../virchow2/MMCI B20-24 Val" \
        --tiles test_heatmaps.csv \
        --morphology slide_list_balanced.csv
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from scipy import stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pt-dir", required=True)
    ap.add_argument("--tiles", required=True, help="heatmap CSV")
    ap.add_argument("--morphology", required=True, help="CSV with slide_name + morphology")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    # Load morphology map
    morph_df = pd.read_csv(args.morphology)
    morph_map = dict(zip(morph_df["slide_name"], morph_df["morphology"]))
    print(f"Morphology: {morph_df['morphology'].value_counts().to_dict()}")

    # Load tile filter
    tiles_df = pd.read_csv(args.tiles)
    tile_slides = set(tiles_df["slide_name"])
    print(f"Tiles: {len(tiles_df)} from {len(tile_slides)} slides")

    # Load embeddings
    pt_dir = Path(args.pt_dir)
    pt_files = sorted(p for p in pt_dir.glob("*.pt") if p.stem in tile_slides)
    print(f"Loading {len(pt_files)} .pt files...")

    all_embs = []
    all_labels = []  # 0 = 8500, 1 = 8520
    t0 = time.monotonic()
    for pt_file in tqdm(pt_files, desc="loading", unit="file"):
        slide_name = pt_file.stem
        if slide_name not in morph_map:
            continue
        label = 0 if str(morph_map[slide_name]) == "8500" else 1

        emb = torch.load(pt_file, map_location="cpu", weights_only=False)
        if isinstance(emb, dict):
            for k in ("embeddings", "embedding", "emb", "features"):
                if k in emb:
                    emb = emb[k]
                    break
        emb = emb.numpy().astype(np.float32)

        # Filter to carcinoma tiles
        slide_tiles = tiles_df[tiles_df["slide_name"] == slide_name]
        if slide_tiles.empty:
            continue
        n_keep = min(emb.shape[0], len(slide_tiles))
        emb = emb[:n_keep]

        all_embs.append(emb)
        all_labels.extend([label] * emb.shape[0])

    X = np.vstack(all_embs)
    y = np.array(all_labels)
    print(f"Loaded: {X.shape[0]} tiles ({time.monotonic()-t0:.0f}s)")
    print(f"  8500: {(y==0).sum()}, 8520: {(y==1).sum()}")

    # L2 normalize
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1
    X /= norms

    # --- Method 1: Mean direction ---
    mean_8500 = X[y == 0].mean(axis=0)
    mean_8520 = X[y == 1].mean(axis=0)
    direction = mean_8520 - mean_8500
    direction /= np.linalg.norm(direction)

    # Project
    proj = X @ direction
    proj_8500 = proj[y == 0]
    proj_8520 = proj[y == 1]

    print(f"\n--- Mean direction projection ---")
    print(f"  8500: mean={proj_8500.mean():.6f}, std={proj_8500.std():.6f}")
    print(f"  8520: mean={proj_8520.mean():.6f}, std={proj_8520.std():.6f}")
    t_stat, p_value = stats.ttest_1samp(proj_8520 - proj_8500, 0) if len(proj_8500) == len(proj_8520) else (stats.ttest_ind(proj_8500, proj_8520))
    print(f"  t-test: t={t_stat:.3f}, p={p_value:.2e}")
    # Overlap: what % of 8520 are above the 8500 mean?
    threshold = proj_8500.mean()
    pct_8520_above = (proj_8520 > threshold).mean()
    pct_8500_above = (proj_8500 > threshold).mean()
    print(f"  Above 8500 mean: 8500={pct_8500_above:.1%}, 8520={pct_8520_above:.1%}")

    # --- Method 2: LDA (proper supervised) ---
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    print(f"\n--- LDA ---")
    t0 = time.monotonic()
    # Subsample for speed (LDA is O(n*d^2))
    n_sub = min(100_000, X.shape[0])
    idx = np.random.choice(X.shape[0], n_sub, replace=False)
    lda = LinearDiscriminantAnalysis()
    lda.fit(X[idx], y[idx])
    # Project all
    proj_lda = lda.transform(X)
    proj_lda_8500 = proj_lda[y == 0]
    proj_lda_8520 = proj_lda[y == 1]
    print(f"  8500: mean={proj_lda_8500.mean():.6f}, std={proj_lda_8500.std():.6f}")
    print(f"  8520: mean={proj_lda_8520.mean():.6f}, std={proj_lda_8520.std():.6f}")
    t_res = stats.ttest_ind(proj_lda_8500, proj_lda_8520)
    print(f"  t-test: t={np.asarray(t_res.statistic).item():.3f}, p={np.asarray(t_res.pvalue).item():.2e}")
    threshold = proj_lda_8500.mean()
    print(f"  Above 8500 mean: 8500={(proj_lda_8500 > threshold).mean():.1%}, 8520={(proj_lda_8520 > threshold).mean():.1%}")
    print(f"  ({time.monotonic()-t0:.0f}s)")

    # --- Summary ---
    print(f"\n--- Conclusion ---")
    # Effect size (Cohen's d) for LDA projection
    d = (proj_lda_8520.mean() - proj_lda_8500.mean()) / np.sqrt((proj_lda_8500.std()**2 + proj_lda_8520.std()**2) / 2)
    print(f"  Cohen's d (LDA): {d:.4f}")
    if abs(d) < 0.2:
        print("  -> Negligible difference. Morphology is NOT encoded in tile embeddings.")
    elif abs(d) < 0.5:
        print("  -> Small difference. Weak signal.")
    elif abs(d) < 0.8:
        print("  -> Medium difference. Some signal.")
    else:
        print("  -> Large difference. Strong signal.")

    # --- Save LDA projections to CSV ---
    # Need tile coords: reload from tiles_df
    # proj_lda is in the same order as X, which was built from pt_files in order
    # We need to map back to (slide_name, x, y)
    # Rebuild the mapping
    out_csv = Path("lda_scores.csv")
    rows = []
    idx = 0
    for pt_file in pt_files:
        slide_name = pt_file.stem
        if slide_name not in morph_map:
            continue
        slide_tiles = tiles_df[tiles_df["slide_name"] == slide_name]
        if slide_tiles.empty:
            continue
        n_keep = min(slide_tiles.shape[0], slide_tiles.shape[0])  # all tiles
        # We need the original embedding size to know how many rows
        # Actually we stored n_keep = min(emb.shape[0], len(slide_tiles)) during loading
        # Let's just use the tiles_df coords directly
        for _, trow in slide_tiles.iterrows():
            if idx < len(proj_lda):
                rows.append((slide_name, int(trow["x"]), int(trow["y"]), float(proj_lda[idx])))
                idx += 1
    # Write CSV
    pd.DataFrame(rows, columns=["slide_name", "x", "y", "value"]).to_csv(out_csv, index=False)
    print(f"\nSaved {len(rows)} tiles to {out_csv}")


if __name__ == "__main__":
    main()
