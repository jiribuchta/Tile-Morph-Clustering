"""Cluster tiles directly with KMeans. Sweep k, report silhouette.

Usage:
    python cluster_tiles.py \
        --pt-dir "/mnt/.../virchow2/MMCI B20-24 Val" \
        --slides slide_list_balanced.csv \
        --morphology slide_morphology.csv \
        --ks 2,4,8,16,32
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pt-dir", required=True, help="dir with per-slide .pt files")
    ap.add_argument("--slides", help="CSV with slide_name column (subset)")
    ap.add_argument("--morphology", help="CSV with slide_name + morphology")
    ap.add_argument("--ks", default="2,4,8,16,32", help="comma-separated k values")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="tile_clustering")
    args = ap.parse_args()

    ks = [int(k) for k in args.ks.split(",")]

    # Load slide list
    if args.slides:
        slides_df = pd.read_csv(args.slides)
        target_names = set(slides_df["slide_name"])
        print(f"Target: {len(target_names)} slides")
    else:
        target_names = None

    # Load all tile embeddings
    pt_dir = Path(args.pt_dir)
    pt_files = sorted(pt_dir.glob("*.pt"))
    if target_names:
        pt_files = [p for p in pt_files if p.stem in target_names]
        print(f"{len(pt_files)} .pt files to load")

    all_embs = []
    all_names = []
    t0 = time.monotonic()
    for i, pt_file in enumerate(pt_files):
        emb = torch.load(pt_file, map_location="cpu", weights_only=True)
        if isinstance(emb, dict):
            for k in ("embeddings", "embedding", "emb", "features"):
                if k in emb:
                    emb = emb[k]
                    break
        emb = emb.numpy().astype(np.float32)
        all_embs.append(emb)
        all_names.extend([pt_file.stem] * emb.shape[0])
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(pt_files)} ({time.monotonic()-t0:.0f}s)")

    X = np.vstack(all_embs)
    names = np.array(all_names)
    print(f"Total: {X.shape[0]} tiles, dim {X.shape[1]} ({time.monotonic()-t0:.0f}s)")

    # L2 normalize
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1
    X /= norms

    # Load morphology if provided
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
        sil = silhouette_score(X[::100], labels[::100])  # subsample for speed
        inertia = km.inertia_
        print(f"  k={k:3d}: silhouette={sil:.4f}, inertia={inertia:.0f} ({time.monotonic()-t0:.1f}s)")
        results.append((k, sil, inertia))

        # Save labels
        pd.DataFrame({
            "slide_name": names,
            "cluster": labels,
        }).to_parquet(out_dir / f"clusters_k{k}.parquet", index=False)

    # Summary
    print("\nSummary:")
    for k, sil, _ in results:
        print(f"  k={k:3d}: silhouette={sil:.4f}")

    # If morphology provided, check cluster-morphology alignment
    if morph_map:
        print("\nCluster x Morphology (best k):")
        best_k = max(results, key=lambda x: x[1])[0]
        df = pd.read_parquet(out_dir / f"clusters_k{best_k}.parquet")
        df["morphology"] = df["slide_name"].map(morph_map)
        ct = pd.crosstab(df["cluster"], df["morphology"])
        print(ct.to_string())

    print(f"\nSaved to {out_dir}/")


if __name__ == "__main__":
    main()
