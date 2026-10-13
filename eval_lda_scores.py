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
    args = ap.parse_args()
    evaluate_scores(pd.read_csv(args.scores), args.morphology)


if __name__ == "__main__":
    main()
