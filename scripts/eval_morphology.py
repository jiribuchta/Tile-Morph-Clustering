"""Test whether morphology (850*/852*) separates from the tile clustering.

Joins a slide-level morphology CSV to the clustering output
(assignments.parquet: slide_id, cluster, carcinoma) and reports:
  - ARI (cluster vs morphology) — the headline number
  - per-cluster morphology share + mean tile-level carcinoma
  - per-morphology mean carcinoma (does the heatmap differ?)
  - chi-square p-value (independence of cluster x morphology)

Usage:
    uv run python scripts/eval_morphology.py \
        --csv morphology.csv --out clustering/k32 \
        [--slide-col slide_id] [--morph-col morphology]
"""

import argparse

import pandas as pd
from scipy.stats import chi2_contingency
from sklearn.metrics import adjusted_rand_score

KEEP_PREFIXES = ("850", "852")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", required=True, help="clustering dir with assignments.parquet")
    ap.add_argument("--slide-col", default="slide_id", help="CSV column joining slide_id")
    ap.add_argument("--morph-col", default="morphology")
    args = ap.parse_args()

    morph = pd.read_csv(args.csv)
    morph[args.morph_col] = morph[args.morph_col].astype(str)
    morph = morph[morph[args.morph_col].str.startswith(KEEP_PREFIXES)]
    morph = morph.drop_duplicates(args.slide_col)
    print(f"CSV: {len(morph)} slides kept "
          f"({morph[args.morph_col].str[:3].value_counts().to_dict()})")

    a = pd.read_parquet(f"{args.out}/assignments.parquet",
                        columns=["slide_id", "cluster", "carcinoma"])
    j = a.merge(morph[[args.slide_col, args.morph_col]],
                left_on="slide_id", right_on=args.slide_col, how="inner")
    if j.empty:
        print("NO JOIN — sample slide_id from assignments:",
              a["slide_id"].head(3).tolist())
        print("sample from CSV:", morph[args.slide_col].head(3).tolist())
        raise SystemExit("check --slide-col / id format")
    print(f"joined {len(j)} tiles from {j['slide_id'].nunique()} slides")

    # headline: ARI between cluster and morphology
    ari = adjusted_rand_score(j["cluster"], j[args.morph_col])
    print(f"\nARI (cluster vs morphology) = {ari:.4f}")

    # per-cluster table
    g = j.groupby("cluster").agg(
        n_tiles=("slide_id", "size"),
        n_slides=("slide_id", "nunique"),
        share_850=(args.morph_col, lambda s: (s.str.startswith("850")).mean()),
        carcin=("carcinoma", "mean"),
    )
    print("\nper cluster:")
    print(g.round(3).to_string())

    # per-morphology: does the tile-level carcinoma heatmap differ?
    pm = j.groupby(args.morph_col).agg(
        n_tiles=("slide_id", "size"),
        n_slides=("slide_id", "nunique"),
        carcin=("carcinoma", "mean"),
    )
    print("\nper morphology (tile-level carcinoma mean):")
    print(pm.round(3).to_string())

    # significance: cluster x morphology contingency
    ct = pd.crosstab(j["cluster"], j[args.morph_col])
    chi2, p, dof, _ = chi2_contingency(ct)
    print(f"\nchi2={chi2:.1f} dof={dof} p={p:.2e}")
    print("interpretation: ARI ~0 => clusters ignore morphology; "
          "high ARI + low p => clusters track 850 vs 852")


if __name__ == "__main__":
    main()
