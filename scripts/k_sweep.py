"""k-sweep on a saved X_norm.npy (no I/O, in-memory KMeans refit).

Usage:
    uv run python scripts/k_sweep.py --out clustering/k32
    uv run python scripts/k_sweep.py --out clustering/k32 --ks 16 32 64 128
    uv run python scripts/k_sweep.py --out clustering/k32 --join-carcin

Rows of X_norm.npy are aligned to assignments.parquet in the same out dir, so
--join-carcin reports per-k carcinoma separability (max cluster mean spread)
for free.
"""

import argparse
import time

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# ponytail: silhouette on 5000 subsample, same as cluster_tiles.py;
# exact silhouette at N=465k is O(N^2) and meaningless in 2560-d anyway.
SIL_N = 5000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="dir with X_norm.npy (and assignments.parquet)")
    ap.add_argument("--ks", type=int, nargs="+", default=[8, 16, 32, 64, 128])
    ap.add_argument("--n-init", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--join-carcin", action="store_true",
                    help="also report per-k mean-carcinoma spread (needs assignments.parquet)")
    args = ap.parse_args()

    X = np.load(f"{args.out}/X_norm.npy")
    carcin = None
    if args.join_carcin:
        import pandas as pd
        a = pd.read_parquet(f"{args.out}/assignments.parquet", columns=["carcinoma"])
        carcin = a["carcinoma"].to_numpy()
        assert len(carcin) == len(X), "assignments.parquet rows != X rows; out dir mismatch?"

    print(f"X: {X.shape} f{X.dtype}")
    print(f"{'k':>4} {'n_pts':>8} {'inertia':>12} {'sil(sub)':>10} "
          f"{'max_cl':>8} {'min_cl':>8} {'med_cl':>8} {'time':>6}")
    t0 = time.monotonic()
    for k in sorted(args.ks):
        tk = time.monotonic()
        km = KMeans(n_clusters=k, n_init=args.n_init, random_state=args.seed).fit(X)
        lab = km.labels_
        if k > 2:
            idx = np.random.default_rng(args.seed).choice(len(lab), min(SIL_N, len(lab)), replace=False)
            sil = silhouette_score(X[idx], lab[idx])
        else:
            sil = float("nan")
        sizes = np.bincount(lab, minlength=k)
        spread = ""
        if carcin is not None:
            means = [float(carcin[lab == i].mean()) for i in range(k) if sizes[i] > 0]
            spread = f"  carcin_range={min(means):.2f}-{max(means):.2f}"
        print(f"{k:>4} {len(X):>8} {km.inertia_:>12.1f} {sil:>10.4f} "
              f"{sizes.max():>8} {sizes[sizes > 0].min():>8} {int(np.median(sizes[sizes > 0])):>8} "
              f"{time.monotonic() - tk:>6.0f}s{spread}")
    print(f"total {time.monotonic() - t0:.0f}s")


if __name__ == "__main__":
    main()
