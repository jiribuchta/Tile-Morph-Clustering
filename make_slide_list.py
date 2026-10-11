"""Join slides.parquet + morphology CSV -> record,path,morphology CSV.

Usage:
    python make_slide_list.py slides.parquet morphology.csv --out slide_list.csv
"""
import argparse
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("slides_parquet", help="slides.parquet with id + path columns")
    ap.add_argument("morphology_csv", help="CSV with record + morphology columns")
    ap.add_argument("--out", default="slide_list.csv")
    args = ap.parse_args()

    slides = pd.read_parquet(args.slides_parquet)
    morph = pd.read_csv(args.morphology_csv, encoding="utf-8-sig", sep=";", engine="python", on_bad_lines="skip")

    # extract slide name from path, strip -NN-X suffix to match morphology record
    slides = slides.copy()
    slides["slide_name"] = slides["path"].apply(
        lambda p: p.rstrip("/").split("/")[-2] if "/" in p else p
    )
    # 2020_00397-01-N -> 2020_00397 (strip -NN-X suffix)
    slides["record"] = slides["slide_name"].apply(lambda s: s.split("-")[0] if "-" in s else s)
    print(f"  slides.record samples: {slides['record'].head(3).tolist()}")
    print(f"  morph.record samples: {morph['record'].head(3).tolist()}")
    merged = slides[["record", "path", "slide_name"]].merge(
        morph[["record", "morphology"]],
        on="record", how="inner",
    )
    merged = merged[["record", "path", "morphology"]]
    merged.to_csv(args.out, index=False)
    print(f"wrote {len(merged)} slides -> {args.out}")
    print(f"  morphology counts:\n{merged['morphology'].value_counts().to_string()}")


if __name__ == "__main__":
    main()
