"""Pick the most confidently classified slide + extreme tiles per morphology.

Prints, for each class (8500 / 8520), the Val slide with the most extreme
mean LDA score (>= 1000 tiles) and its 3 most extreme tiles as x, y
coordinates (level-1 pixels, same grid as the xOpat masks) — for jumping
straight to textbook examples in the viewer.

Usage:
    python pick_example_tiles.py --scores lda_scores_v5_095_val.csv \
        --morphology /mnt/projects/breast_cancer/bc/dataframe.csv
"""
import argparse

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", required=True, help="lda_scores CSV (slide_name,x,y,value)")
    ap.add_argument("--morphology", required=True, help="CSV with slide_id + morphology")
    ap.add_argument("--min-tiles", type=int, default=1000, help="min tiles for a slide to qualify")
    ap.add_argument("--top", type=int, default=3, help="tiles to list per slide")
    args = ap.parse_args()

    m = pd.read_csv(args.morphology)
    m["morphology"] = m["morphology"].astype(str).str.replace(" ", "")
    label = dict(zip(m["slide_id"], m["morphology"], strict=True))
    df = pd.read_csv(args.scores)
    df["morph"] = df["slide_name"].map(label)
    df = df.dropna(subset=["morph"])

    for code, worst in [("8500", True), ("8520", False)]:
        sub = df[df["morph"] == code]
        st = sub.groupby("slide_name")["value"].agg(["mean", "size"])
        st = st[st["size"] >= args.min_tiles]
        if st.empty:
            print(f"{code}: no slide with >= {args.min_tiles} tiles")
            continue
        slide = st["mean"].idxmin() if worst else st["mean"].idxmax()
        print(f"\n{code}: best slide {slide} "
              f"(mean {st.loc[slide, 'mean']:.2f}, {int(st.loc[slide, 'size'])} tiles)")
        t = sub[sub["slide_name"] == slide]
        t = t.nsmallest(args.top, "value") if worst else t.nlargest(args.top, "value")
        print(t[["x", "y", "value"]].to_string(index=False))


if __name__ == "__main__":
    main()
