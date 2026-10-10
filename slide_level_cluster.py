"""Cluster at slide level: average region embeddings per slide, k-means, report vs morphology.

Uses the cached X_regions.npy + region_meta from cluster_regions.py.
    python slide_level_cluster.py --out /mnt/projects/breast_cancer/tile_morph_clustering/region_clustering_1 \
        --morphology /mnt/projects/breast_cancer/tile_morph_clustering/slide_morphology.csv --k 3
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--morphology", required=True)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out = Path(args.out)
    X = np.load(out / "X_regions.npy")
    if (out / "region_meta.parquet").exists():
        meta = pd.read_parquet(out / "region_meta.parquet")
    else:
        meta = pd.read_parquet(out / "regions.parquet")[["slide_id", "region", "n_tiles", "cx", "cy"]]
    morph = pd.read_csv(args.morphology)
    morph_by_sid = dict(zip(morph["slide_name"], morph["morphology"]))

    # average region embeddings per slide (weighted by n_tiles)
    slide_ids = meta["slide_id"].values
    weights = meta["n_tiles"].values.astype(np.float64)
    unique_slides = np.unique(slide_ids)
    X_slides = np.stack([
        (X[slide_ids == s] * weights[slide_ids == s][:, None]).sum(axis=0) / weights[slide_ids == s].sum()
        for s in unique_slides
    ])
    norms = np.linalg.norm(X_slides, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    X_slides /= norms
    print(f"slide-level embeddings: {X_slides.shape}")

    km = KMeans(n_clusters=args.k, n_init=4, random_state=args.seed).fit(X_slides)
    labels = km.labels_
    idx = np.random.default_rng(args.seed).choice(len(labels), size=min(len(labels), args.k * 100), replace=False)
    sil = silhouette_score(X_slides[idx], labels[idx])
    print(f"  inertia={km.inertia_:.1f}  silhouette(subsample)={sil:.3f}")

    df = pd.DataFrame({"slide_id": unique_slides, "cluster": labels})
    df["morphology"] = df["slide_id"].map(morph_by_sid)
    ct = (
        df.dropna(subset=["morphology"])
        .groupby(["morphology", "cluster"])
        .size()
        .unstack(fill_value=0)
    )
    print("\nmorphology vs cluster (slides):")
    print(ct)

    summary = {
        "k": args.k,
        "n_slides": len(unique_slides),
        "n_slides_labeled": df["morphology"].notna().sum(),
        "silhouette_subsample": round(float(sil), 4),
        "morphology_vs_cluster": ct.reset_index().to_dict("records"),
    }
    dest = out / "slide_level_summary.json"
    dest.write_text(json.dumps(summary, indent=2))
    print(f"\nwrote: {dest}")


if __name__ == "__main__":
    main()
