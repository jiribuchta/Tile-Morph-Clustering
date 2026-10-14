"""Evaluate an existing LDA scores CSV against morphology (no re-scoring).

Usage:
    python eval_lda_scores.py --scores lda_scores.csv --morphology dataframe.csv
"""

import argparse

import pandas as pd

from lda_projection import evaluate_scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", required=True, help="CSV: slide_name, x, y, value")
    ap.add_argument("--morphology", required=True, help="CSV with slide_name/slide_id + morphology")
    ap.add_argument("--min-tiles", type=int, default=100,
                    help="exclude slides with fewer scored tiles (their mean is noise)")
    args = ap.parse_args()
    evaluate_scores(pd.read_csv(args.scores), args.morphology, args.min_tiles)


if __name__ == "__main__":
    main()
