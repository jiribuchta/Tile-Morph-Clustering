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
    morph = pd.read_csv(args.morphology_csv, encoding="utf-8-sig", engine="python", on_bad_lines="skip")
    print(f"morphology CSV columns: {list(morph.columns)}")

    # join on id == record
    merged = slides[["id", "path"]].merge(
        morph[["record", "morphology"]],
        left_on="id", right_on="record", how="inner",
    )
    merged = merged[["record", "path", "morphology"]]
    merged.to_csv(args.out, index=False)
    print(f"wrote {len(merged)} slides -> {args.out}")
    print(f"  morphology counts:\n{merged['morphology'].value_counts().to_string()}")


if __name__ == "__main__":
    main()
