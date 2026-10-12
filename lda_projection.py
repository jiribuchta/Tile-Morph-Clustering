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
    ap.add_argument("--out", default="lda_scores.csv", help="output CSV path")
    ap.add_argument("--save-direction", default=None, help="save LDA direction .npy (for scoring new slides)")
    ap.add_argument("--load-direction", default=None, help="load LDA direction .npy to score new slides (skip training)")
    args = ap.parse_args()

    # --- Load-direction mode: score new slides with a saved direction ---
    if args.load_direction:
        direction = np.load(args.load_direction)
        print(f"Loaded LDA direction ({direction.shape[0]} dims) from {args.load_direction}")

        tiles_df = pd.read_csv(args.tiles)
        pt_dir = Path(args.pt_dir)
        pt_files = sorted(pt_dir.glob("*.pt"))
        print(f"Scoring {len(pt_files)} slides...")

        rows = []
        for pt_file in tqdm(pt_files, desc="scoring", unit="slide"):
            slide_name = pt_file.stem
            slide_tiles = tiles_df[tiles_df["slide_name"] == slide_name]
            if slide_tiles.empty:
                continue
            emb = torch.load(pt_file, map_location="cpu", weights_only=False)
            if isinstance(emb, dict):
                for k in ("embeddings", "embedding", "emb", "features"):
                    if k in emb:
                        emb = emb[k]
                        break
            emb = emb.numpy().astype(np.float32)
            n_keep = min(emb.shape[0], len(slide_tiles))
            emb = emb[:n_keep]
            # L2 normalize + project
            norms = np.linalg.norm(emb, axis=1, keepdims=True)
            norms[norms == 0] = 1
            emb /= norms
            scores = emb @ direction
            for i, (_, trow) in enumerate(slide_tiles.iterrows()):
                rows.append((slide_name, int(trow["x"]), int(trow["y"]), scores[i].item()))

        pd.DataFrame(rows, columns=["slide_name", "x", "y", "value"]).to_csv(args.out, index=False)
        print(f"Saved {len(rows)} tiles to {args.out}")
        return

    # --- Training mode ---
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

    # --- Save LDA direction ---
    if args.save_direction:
        # LDA direction = coef_ * scaled_intercept_ logic
        # For LDA, the projection direction is lda.coef_ (shape: [n_classes, n_features])
        # For 2 classes, the separating direction is coef_[1] - coef_[0]
        direction = lda.coef_[1] - lda.coef_[0]
        direction /= np.linalg.norm(direction)
        np.save(args.save_direction, direction)
        print(f"\nSaved LDA direction to {args.save_direction}")

    # --- Save LDA projections to CSV ---
    out_csv = Path(args.out)
    rows = []
    idx = 0
    for pt_file in pt_files:
        slide_name = pt_file.stem
        if slide_name not in morph_map:
            continue
        slide_tiles = tiles_df[tiles_df["slide_name"] == slide_name]
        if slide_tiles.empty:
            continue
        for _, trow in slide_tiles.iterrows():
            if idx < len(proj_lda):
                rows.append((slide_name, int(trow["x"]), int(trow["y"]), proj_lda[idx].item()))
                idx += 1
    pd.DataFrame(rows, columns=["slide_name", "x", "y", "value"]).to_csv(out_csv, index=False)
    print(f"\nSaved {len(rows)} tiles to {out_csv}")


if __name__ == "__main__":
    main()
