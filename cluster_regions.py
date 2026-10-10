"""Unsupervised clustering of carcinoma *regions* (connected blobs of heatmap tiles).

Per slide: cancerous tiles (value >= min_value) from heatmaps_tiles.csv are labeled
into 8-connected regions (scipy.ndimage.label). Each region gets one feature vector
= mean of its tiles' L2-normalized Virchow2 embeddings. KMeans on regions, then the
cluster-vs-morphology (8500/8520) cross-tab is reported.

Usage:
    python cluster_regions.py \
        --tiles heatmaps_tiles.csv \
        --morphology slide_morphology.csv \
        --embeddings "/path/to/MMCI B20-24 Train_sharded" \
        --min-value 0.5 --min-region 8 --k 2

--embeddings: sharded dir with tiles/*.parquet (and slides/slides.parquet).
Writes: regions.parquet, X_regions.npy, centroids.npy, summary.json
"""
import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import tifffile
from scipy import ndimage
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

ENC = 1_000_000  # x*ENC+y is unique for these coords


def load_regions(tiles_csv, min_value, min_region):
    """slide_id -> dict(encs=sorted int64, labels=int32 per enc, regions=[(n, cx, cy)])"""
    per_slide: dict[str, list[tuple[int, int, float]]] = {}
    names: dict[str, str] = {}
    with open(tiles_csv, newline="") as f:
        for row in csv.DictReader(f):
            v = float(row["value"])
            if v >= min_value:
                per_slide.setdefault(row["slide_id"], []).append(
                    (int(row["x"]) * ENC + int(row["y"]), int(row["x"]), int(row["y"]))
                )
                names.setdefault(row["slide_id"], row["slide_name"])

    out = {}
    for sid, tiles in per_slide.items():
        encs = np.array([t[0] for t in tiles], dtype=np.int64)
        xs = np.array([t[1] for t in tiles], dtype=np.int32)
        ys = np.array([t[2] for t in tiles], dtype=np.int32)
        cols, rows = xs // 112, ys // 112
        H, W = rows.max() + 1, cols.max() + 1
        mask = np.zeros((H, W), dtype=bool)
        mask[rows, cols] = True
        xs2d = np.zeros((H, W), dtype=np.int32)
        ys2d = np.zeros((H, W), dtype=np.int32)
        xs2d[rows, cols] = xs
        ys2d[rows, cols] = ys
        lab, n = ndimage.label(mask, structure=np.ones((3, 3), dtype=int))
        counts = np.bincount(lab.ravel(), minlength=n + 1)
        keep = np.flatnonzero(counts[1:] >= min_region) + 1  # region ids to keep (0 = background)
        remap = np.zeros(n + 1, dtype=np.int32)
        for new, old in enumerate(keep, start=1):
            remap[old] = new
        labels = remap[lab[rows, cols]].astype(np.int32)
        grid = np.zeros((H, W), dtype=np.int32)
        grid[rows, cols] = labels
        order = np.argsort(encs)
        encs, labels = encs[order], labels[order]
        regions = []
        for rid in keep:
            m = lab == rid
            regions.append((int(m.sum()), float(xs2d[m].mean()), float(ys2d[m].mean())))
        out[sid] = {"encs": encs, "labels": labels, "regions": regions, "grid": grid, "name": names[sid]}
    return out


def index_parts(embed_dir):
    """slide_id(hex) -> part path, using only the slide_id column."""
    tiles_dir = Path(embed_dir) / "tiles" if (Path(embed_dir) / "tiles").is_dir() else Path(embed_dir)
    parts = sorted(tiles_dir.glob("*.parquet"))
    slide_to_part: dict[str, str] = {}
    for p in parts:
        pf = pq.ParquetFile(str(p))
        for batch in pf.iter_batches(columns=["slide_id"], batch_size=65536):
            for sid in batch.column("slide_id").to_numpy(zero_copy_only=False):
                h = sid.hex() if isinstance(sid, (bytes, bytearray)) else str(sid)
                slide_to_part.setdefault(h, str(p))
    return slide_to_part, parts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiles", default="heatmaps_tiles.csv")
    ap.add_argument("--morphology", default="slide_morphology.csv")
    ap.add_argument("--embeddings", help="sharded dir with tiles/*.parquet (not needed with --load-x)")
    ap.add_argument("--min-value", type=float, default=0.5)
    ap.add_argument("--min-region", type=int, default=8, help="drop regions smaller than this many tiles")
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="out_regions")
    ap.add_argument("--load-x", action="store_true",
                    help="load X_regions.npy + region_meta.parquet from --out, skip embedding collection")
    args = ap.parse_args()
    if not args.load_x and not args.embeddings:
        raise SystemExit("--embeddings required unless --load-x")
    t0 = time.monotonic()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # morphology: slide_name -> morphology (, or tab separated)
    import pandas as pd
    with open(args.morphology) as f:
        sep = "\t" if "\t" in f.readline() else ","
    morph = pd.read_csv(args.morphology, sep=sep)
    morph_by_name = dict(zip(morph["slide_name"].str.strip(), morph["morphology"].astype(str).str.strip()))

    print("loading carcinoma regions from heatmap CSV...")
    regions = load_regions(args.tiles, args.min_value, args.min_region)
    n_tiles = sum(len(r["encs"]) for r in regions.values())
    n_regions = sum(len(r["regions"]) for r in regions.values())
    print(f"  {len(regions)} slides, {n_tiles} carcinoma tiles, {n_regions} regions")

    # slide_id -> morphology (via slide_name in the tiles CSV)
    morph_by_sid: dict[str, str] = {}
    with open(args.tiles, newline="") as f:
        for row in csv.DictReader(f):
            nm = row["slide_name"]
            if nm in morph_by_name:
                morph_by_sid.setdefault(row["slide_id"], morph_by_name[nm])
    print(f"  {len(morph_by_sid)} slides have a morphology label")

    if args.load_x:
        print(f"loading cached X from {out} (--load-x)")
        X = np.load(out / "X_regions.npy")
        # region_meta.parquet (new runs) or regions.parquet (older runs, same row order)
        if (out / "region_meta.parquet").exists():
            meta = pd.read_parquet(out / "region_meta.parquet")
        else:
            meta = pd.read_parquet(out / "regions.parquet")[
                ["slide_id", "region", "n_tiles", "cx", "cy"]
            ]
        region_meta = meta.to_dict("records")
        matched = int(meta["n_tiles"].sum())
    else:
        print("indexing embedding parts (slide_id column only)...")
        slide_to_part, all_parts = index_parts(args.embeddings)
        target_parts = sorted({slide_to_part[s] for s in regions if s in slide_to_part})
        missing = [s for s in regions if s not in slide_to_part]
        print(f"  {len(all_parts)} parts total, {len(target_parts)} needed for our slides")
        if missing:
            print(f"  [warn] {len(missing)} slides not found in embeddings, e.g. {missing[:5]}")

        # accumulate per-region embedding sums
        sums: dict[int, np.ndarray] = {}
        counts: dict[int, int] = {}
        region_meta: list[dict] = []  # one row per region, index == region key
        for sid, r in regions.items():
            for i, (n, cx, cy) in enumerate(r["regions"], start=1):
                key = (sid, i)
                sums[key] = np.zeros(2560, dtype=np.float32)
                counts[key] = 0
                region_meta.append({"slide_id": sid, "region": i, "n_tiles": n, "cx": cx, "cy": cy})

        print(f"collecting embeddings from {len(target_parts)} parts...")
        matched = 0
        for pi, part in enumerate(target_parts):
            pf = pq.ParquetFile(part)
            for batch in pf.iter_batches(columns=["slide_id", "x", "y", "embedding"], batch_size=4096):
                sids = batch.column("slide_id").to_numpy(zero_copy_only=False)
                xs = batch.column("x").to_numpy(zero_copy_only=False).astype(np.int64)
                ys = batch.column("y").to_numpy(zero_copy_only=False).astype(np.int64)
                emb = batch.column("embedding").values.to_numpy(zero_copy_only=False)
                n = len(batch)
                if emb.size % n:
                    raise SystemExit("ragged embeddings; not supported")
                emb = emb.reshape(n, -1).astype(np.float32)
                if emb.shape[1] != 2560:
                    raise SystemExit(f"unexpected embedding dim {emb.shape[1]} (expected 2560)")
                # L2 normalize rows (in place)
                norms = np.linalg.norm(emb, axis=1, keepdims=True)
                norms[norms == 0] = 1.0
                emb /= norms
                enc = xs * ENC + ys
                sids_h = np.array(
                    [s.hex() if isinstance(s, (bytes, bytearray)) else str(s) for s in sids]
                )
                codes, inv = np.unique(sids_h, return_inverse=True)
                for ci, h in enumerate(codes):
                    r = regions.get(h)
                    if r is None:
                        continue
                    idx = np.flatnonzero(inv == ci)
                    j = np.searchsorted(r["encs"], enc[idx])
                    ok = j < len(r["encs"])
                    j, idx = j[ok], idx[ok]
                    ok = r["encs"][j] == enc[idx]
                    idx, j = idx[ok], j[ok]
                    rl = r["labels"][j]
                    keep = rl > 0  # label 0 = tile in a dropped (too-small) region
                    idx, rl = idx[keep], rl[keep]
                    if not len(idx):
                        continue
                    for rid in np.unique(rl):
                        m = rl == rid
                        key = (h, int(rid))
                        sums[key] += emb[idx[m]].sum(axis=0)
                        counts[key] += int(m.sum())
                        matched += int(m.sum())
            if (pi + 1) % 10 == 0 or pi == len(target_parts) - 1:
                print(f"  part {pi+1}/{len(target_parts)} ({time.monotonic()-t0:.0f}s)")
        print(f"  matched {matched} tiles into regions")

        n_before = len(region_meta)
        region_meta = [m for m in region_meta if counts[(m["slide_id"], m["region"])] > 0]
        print(f"  {n_before - len(region_meta)} regions dropped (no matching embedding tiles)")
        if not region_meta:
            raise SystemExit("no tiles matched any region; check slide_id format / paths")
        X = np.stack([sums[(m["slide_id"], m["region"])] / counts[(m["slide_id"], m["region"])]
                      for m in region_meta])
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        X /= norms
        np.save(out / "X_regions.npy", X)
        pd.DataFrame(region_meta).to_parquet(out / "region_meta.parquet")

    print(f"fitting KMeans k={args.k} on {X.shape}...")
    km = KMeans(n_clusters=args.k, n_init=4, random_state=args.seed).fit(X)
    labels = km.labels_
    rng = np.random.default_rng(args.seed)
    idx = rng.choice(len(labels), size=min(3000, len(labels)), replace=False)
    sil = silhouette_score(X[idx], labels[idx]) if len(idx) > args.k else float("nan")
    print(f"  inertia={km.inertia_:.1f}  silhouette(subsample)={sil:.3f}")
    np.save(out / "centroids.npy", km.cluster_centers_.astype(np.float32))

    # tiff mask per slide: 0 = background, cluster c -> c+1
    cluster_by_key = {(m["slide_id"], m["region"]): c for m, c in zip(region_meta, labels)}
    masks_dir = out / "masks"
    masks_dir.mkdir(exist_ok=True)
    n_masks = 0
    for sid, r in regions.items():
        g = r["grid"]
        if g.max() == 0:
            continue
        lut = np.zeros(g.max() + 1, dtype=np.uint8)
        for i in range(1, len(r["regions"]) + 1):
            c = cluster_by_key.get((sid, i))
            if c is not None:
                lut[i] = c + 1
        name = r["name"].replace("/", "_")
        tifffile.imwrite(masks_dir / f"{name}.tiff", lut[g])
        n_masks += 1
    print(f"  wrote {n_masks} tiff masks to {masks_dir}")

    # per-slide majority cluster vs morphology
    df = pd.DataFrame(region_meta)
    df["cluster"] = labels
    df["morphology"] = df["slide_id"].map(morph_by_sid)
    df.to_parquet(out / "regions.parquet")

    per_slide = (
        df.groupby(["slide_id", "cluster"]).size().reset_index(name="n")
        .sort_values("n")
        .groupby("slide_id")
        .tail(1)
        .rename(columns={"cluster": "majority_cluster"})
    )
    per_slide["morphology"] = per_slide["slide_id"].map(morph_by_sid)
    ct = (
        per_slide.groupby(["morphology", "majority_cluster"])
        .size()
        .unstack(fill_value=0)
    )
    summary = {
        "k": args.k,
        "n_regions": len(df),
        "n_slides": df["slide_id"].nunique(),
        "n_tiles_matched": matched,
        "min_value": args.min_value,
        "min_region": args.min_region,
        "silhouette_subsample": round(float(sil), 4),
        "morphology_vs_cluster": ct.reset_index().to_dict("records"),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))

    # report config (never fatal)
    tpl_path = Path(__file__).parent / "report_conf" / "reporter" / "region_clustering.yaml"
    if tpl_path.exists():
        tpl = tpl_path.read_text()
        tpl = tpl.replace(
            "dir_name: /mnt/projects/breast_cancer/tile_morph_clustering/region_clustering_1/masks",
            f"dir_name: {masks_dir.resolve()}",
        )
        # restrict background to only slides that have masks
        slide_files = [f"{r['name'].replace('/', '_')}.mrxs" for r in regions.values() if r["grid"].max() > 0]
        globs_block = "\n".join(f'  - "{f}"' for f in sorted(slide_files))
        tpl = tpl.replace('globs: ["*.mrxs"]', f"globs:\n{globs_block}")
        dest = out / "report_conf" / "reporter" / "region_clustering.yaml"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(tpl)
        print(f"wrote report config: {dest} ({len(slide_files)} slides)")

    print(f"\nmorphology vs majority cluster (slides):")
    print(ct.to_string())
    print(f"wrote: {out} ({time.monotonic()-t0:.0f}s total)")


if __name__ == "__main__":
    main()
