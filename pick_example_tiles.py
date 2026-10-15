"""Pick example slides/tiles from LDA scores — two modes.

Default: per class (8500 / 8520), the Val slide with the most extreme
mean LDA score (>= min tiles) and its most extreme tiles as x, y
coordinates (level-1 pixels, same grid as the xOpat masks).

--mixed: "50/50" slides — well-sampled slides whose tiles split closest to
half ductal (score < 0) / half lobular (score > 0); mixed-tumor candidates.
For each, the most extreme blue and red tile are listed for side-by-side
comparison in the viewer.

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
    ap.add_argument("--mixed", action="store_true", help="find 50/50 slides (mixed-tumor candidates)")
    ap.add_argument("--num", type=int, default=5, help="how many mixed slides to list")
    args = ap.parse_args()

    m = pd.read_csv(args.morphology)
    m["morphology"] = m["morphology"].astype(str).str.replace(" ", "")
    label = dict(zip(m["slide_id"], m["morphology"], strict=True))
    df = pd.read_csv(args.scores)
    df["morph"] = df["slide_name"].map(label)
    df = df.dropna(subset=["morph"])

    if args.mixed:
        g = df.groupby("slide_name")
        st = g["value"].agg(size="size", mean="mean")
        st["pos_frac"] = g["value"].apply(lambda v: (v > 0).mean())
        st = st[st["size"] >= args.min_tiles]
        st["labeled"] = st.index.map(lambda s: label.get(s, "?"))
        best = st.reindex((st["pos_frac"] - 0.5).abs().sort_values().index).head(args.num)
        for slide, row in best.iterrows():
            print(f"\n{slide} (labeled {row['labeled']}) — {int(row['size'])} tiles, "
                  f"{row['pos_frac']:.0%} lobular-leaning, mean {row['mean']:+.2f}")
            t = df[df["slide_name"] == slide]
            print("  bluest tile: ", t.loc[t["value"].idxmin(), ["x", "y", "value"]].to_dict())
            print("  reddest tile:", t.loc[t["value"].idxmax(), ["x", "y", "value"]].to_dict())
        return

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
