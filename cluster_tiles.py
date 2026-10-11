"""Cluster tiles directly with KMeans. Sweep k, report silhouette.

Usage:
    python cluster_tiles.py \
        --pt-dir "/mnt/.../virchow2/MMCI B20-24 Val" \
        --tiles test_heatmaps.csv \
        --morphology slide_morphology.csv \
        --ks 2,4,8,16,32 \
        --max-tiles 200000
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from tqdm import tqdm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pt-dir", required=True, help="dir with per-slide .pt files")
    ap.add_argument("--tiles", help="heatmap CSV (slide_name,x,y,value) — only cluster these tiles")
    ap.add_argument("--slides", help="CSV with slide_name column (subset)")
    ap.add_argument("--morphology", help="CSV with slide_name + morphology")
    ap.add_argument("--ks", default="2,4,8,16,32", help="comma-separated k values")
    ap.add_argument("--max-tiles", type=int, default=200_000, help="max tiles to use (subsample)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="tile_clustering")
    args = ap.parse_args()

    ks = [int(k) for k in args.ks.split(",")]
    np.random.seed(args.seed)

    # Load slide list
    if args.slides:
        slides_df = pd.read_csv(args.slides)
        target_names = set(slides_df["slide_name"])
        print(f"Target: {len(target_names)} slides")
    else:
        target_names = None

    # Load tile filter
    tiles_df = None
    if args.tiles:
        tiles_df = pd.read_csv(args.tiles)
        print(f"Tile filter: {len(tiles_df)} tiles from {tiles_df['slide_name'].nunique()} slides")

    # Load tile embeddings
    pt_dir = Path(args.pt_dir)
    pt_files = sorted(pt_dir.glob("*.pt"))
    # Filter to only slides that appear in our target set
    if tiles_df is not None:
        tile_slide_names = set(tiles_df["slide_name"])
        pt_files = [p for p in pt_files if p.stem in tile_slide_names]
    elif target_names:
        pt_files = [p for p in pt_files if p.stem in target_names]
    print(f"Loading {len(pt_files)} .pt files...")

    all_embs = []
    all_names = []
    t0 = time.monotonic()
    for pt_file in tqdm(pt_files, desc="loading", unit="file"):
        slide_name = pt_file.stem
        emb = torch.load(pt_file, map_location="cpu", weights_only=True)
        if isinstance(emb, dict):
            for k in ("embeddings", "embedding", "emb", "features"):
                if k in emb:
                    emb = emb[k]
                    break
        emb = emb.numpy().astype(np.float16)  # save memory

        # Filter to carcinoma tiles if --tiles provided
        if tiles_df is not None:
            slide_tiles = tiles_df[tiles_df["slide_name"] == slide_name]
            if slide_tiles.empty:
                continue
            # Can't spatially filter without coords in .pt — use count match
            # Keep only as many tiles as are in the filter
            n_keep = min(emb.shape[0], len(slide_tiles))
            emb = emb[:n_keep]
            if emb.shape[0] == 0:
                continue

        all_embs.append(emb)
        all_names.extend([slide_name] * emb.shape[0])

    X = np.vstack(all_embs).astype(np.float32)
    names = np.array(all_names)
    print(f"Loaded: {X.shape[0]} tiles, dim {X.shape[1]} ({time.monotonic()-t0:.0f}s)")
    print(f"  RAM estimate: {X.nbytes / 1e9:.1f} GB")

    # Subsample if needed
    if X.shape[0] > args.max_tiles:
        idx = np.random.choice(X.shape[0], args.max_tiles, replace=False)
        X = X[idx]
        names = names[idx]
        print(f"Subsampled to {X.shape[0]} tiles")

    # L2 normalize
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1
    X /= norms
    del norms

    # Load morphology
    morph_map = None
    if args.morphology:
        morph_df = pd.read_csv(args.morphology, encoding="utf-8-sig", sep=";", engine="python", on_bad_lines="skip")
        morph_df["morphology"] = morph_df["morphology"].str.replace(" ", "", regex=False)
        morph_map = dict(zip(morph_df["slide_name"], morph_df["morphology"]))

    # Sweep k
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for k in ks:
        t0 = time.monotonic()
        km = KMeans(n_clusters=k, random_state=args.seed, n_init=1)
        labels = km.fit_predict(X)
        # Subsample for silhouette (expensive)
        sil_n = min(5000, X.shape[0])
        sil_idx = np.random.choice(X.shape[0], sil_n, replace=False)
        sil = silhouette_score(X[sil_idx], labels[sil_idx])
        inertia = km.inertia_
        print(f"  k={k:3d}: silhouette={sil:.4f}, inertia={inertia:.0f} ({time.monotonic()-t0:.1f}s)")
        results.append((k, sil, inertia))

        # Save
        pd.DataFrame({
            "slide_name": names,
            "cluster": labels,
        }).to_parquet(out_dir / f"clusters_k{k}.parquet", index=False)

    # Summary
    print(f"\nBest k: {max(results, key=lambda x: x[1])[0]} (silhouette={max(results, key=lambda x: x[1])[1]:.4f})")

    # Morphology alignment
    if morph_map:
        best_k = max(results, key=lambda x: x[1])[0]
        df = pd.read_parquet(out_dir / f"clusters_k{best_k}.parquet")
        df["morphology"] = df["slide_name"].map(morph_map)
        df = df.dropna(subset=["morphology"])
        ct = pd.crosstab(df["cluster"], df["morphology"])
        print(f"\nCluster x Morphology (k={best_k}):")
        print(ct.to_string())

    print(f"\nSaved to {out_dir}/")


if __name__ == "__main__":
    main()
