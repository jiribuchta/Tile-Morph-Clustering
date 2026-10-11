"""Join slides.parquet + morphology CSV -> record,path,morphology CSV.

Usage:
    python make_slide_list.py slides.parquet morphology.csv --out slide_list.csv
"""
import argparse
import pandas as pd
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("slides_parquet", help="slides.parquet with id + path columns")
    ap.add_argument("morphology_csv", help="CSV with record + morphology columns")
    ap.add_argument("--out", default="slide_list.csv")
    ap.add_argument("--balanced", type=int, default=None,
                    help="pick N slides per morphology (balanced subset)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    slides = pd.read_parquet(args.slides_parquet)
    morph = pd.read_csv(args.morphology_csv, encoding="utf-8-sig", sep=";", engine="python", on_bad_lines="skip")
    morph["morphology"] = morph["morphology"].str.replace(" ", "", regex=False)

    # extract slide name from path, strip -NN-X suffix to match morphology record
    slides = slides.copy()
    slides["slide_name"] = slides["path"].apply(
        lambda p: p.rstrip("/").split("/")[-2] if "/" in p else p
    )
    # 2020_00397-01-N -> 2020_00397 (strip -NN-X suffix)
    slides["record"] = slides["slide_name"].apply(lambda s: s.split("-")[0] if "-" in s else s)
    print(f"  slides.record samples: {slides['record'].head(3).tolist()}")
    print(f"  morph.record samples: {morph['record'].head(3).tolist()}")
    # keep only T (tumor/positive) samples: suffix after last dash
    slides["suffix"] = slides["slide_name"].apply(lambda s: s.rsplit("-", 1)[-1] if "-" in s else "")
    slides = slides[slides["suffix"].str.upper().str.startswith("T")]
    print(f"  {len(slides)} T (positive) slides")

    merged = slides[["record", "path", "slide_name"]].merge(
        morph[["record", "morphology"]],
        on="record", how="inner",
    )
    merged = merged[["record", "path", "morphology", "slide_name"]]
    merged.to_csv(args.out, index=False)
    print(f"wrote {len(merged)} slides -> {args.out}")
    print(f"  morphology counts:\n{merged['morphology'].value_counts().to_string()}")

    if args.balanced:
        # pick N slides per morphology (8500, 8520 only), output comma-separated
        merged = merged[merged["morphology"].isin(["8500", "8520"])]
        parts = []
        for morph_val, group in merged.groupby("morphology"):
            n = min(args.balanced, len(group))
            parts.append(group.sample(n=n, random_state=args.seed)["slide_name"].tolist())
        balanced_df = merged[merged["slide_name"].isin(all_slides)]
        balanced_csv = args.out.replace(".csv", "_balanced.csv")
        balanced_df.to_csv(balanced_csv, index=False)
        print(f"wrote {len(all_slides)} balanced slides -> {balanced_csv}")
        print(f"  --slides {','.join(all_slides)}")


if __name__ == "__main__":
    main()
