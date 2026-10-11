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
import openslide
import pyarrow.parquet as pq
import pyvips
from ratiopath.masks import write_big_tiff
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
        x_off = int(xs.min()) % 112  # tiling origin offset in level-1 pixels
        y_off = int(ys.min()) % 112
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
        out[sid] = {"encs": encs, "labels": labels, "regions": regions, "grid": grid, "name": names[sid], "x_off": x_off, "y_off": y_off}
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
    ap.add_argument("--data", default=None,
                    help="shared data dir (X_regions.npy, region_meta.parquet). "
                         "Default: same as --out.")
    ap.add_argument("--wsi-root", default="/mnt/bioptic_tree",
                    help="root of bioptic tree for WSI paths")
    ap.add_argument("--load-x", action="store_true",
                    help="load X_regions.npy + region_meta.parquet from --data, skip embedding collection")
    ap.add_argument("--slides", default=None,
                    help="process only this slide name (comma-separated for multiple)")
    args = ap.parse_args()
    if not args.load_x and not args.embeddings:
        raise SystemExit("--embeddings required unless --load-x")
    t0 = time.monotonic()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    data = Path(args.data) if args.data else out
    data.mkdir(parents=True, exist_ok=True)

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
    if args.slides:
        keep = set(args.slides.split(","))
        regions = {k: v for k, v in regions.items() if k in keep}
        if not regions:
            raise SystemExit(f"none of {args.slides} found in regions")
        print(f"  filtered to {len(regions)} slide(s): {sorted(regions)}")

    # slide_id -> morphology (via slide_name in the tiles CSV)
    morph_by_sid: dict[str, str] = {}
    with open(args.tiles, newline="") as f:
        for row in csv.DictReader(f):
            nm = row["slide_name"]
            if nm in morph_by_name:
                morph_by_sid.setdefault(row["slide_id"], morph_by_name[nm])
    print(f"  {len(morph_by_sid)} slides have a morphology label")

    if args.load_x:
        print(f"loading cached X from {data} (--load-x)")
        X = np.load(data / "X_regions.npy")
        # region_meta.parquet (new runs) or regions.parquet (older runs, same row order)
        if (data / "region_meta.parquet").exists():
            meta = pd.read_parquet(data / "region_meta.parquet")
        else:
            meta = pd.read_parquet(data / "regions.parquet")[
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
        np.save(data / "X_regions.npy", X)
        pd.DataFrame(region_meta).to_parquet(data / "region_meta.parquet")

    print(f"fitting KMeans k={args.k} on {X.shape}...")
    km = KMeans(n_clusters=args.k, n_init=4, random_state=args.seed).fit(X)
    labels = km.labels_
    rng = np.random.default_rng(args.seed)
    idx = rng.choice(len(labels), size=min(3000, len(labels)), replace=False)
    sil = silhouette_score(X[idx], labels[idx]) if len(idx) > args.k else float("nan")
    print(f"  inertia={km.inertia_:.1f}  silhouette(subsample)={sil:.3f}")
    np.save(out / "centroids.npy", km.cluster_centers_.astype(np.float32))

    # BigTIFF mask per slide: 0 = background, cluster c -> c+1 (xOpat-ready)
    cluster_by_key = {(m["slide_id"], m["region"]): c for m, c in zip(region_meta, labels)}
    def wsi_path(name: str) -> str:
        year, month, day = name[:4], name[5:7], name[7:10]
        return str(Path(args.wsi_root) / year / month / day / name / f"{name}.mrxs")

    masks_dir = out / "masks"
    masks_dir.mkdir(exist_ok=True)
    n_masks = n_skip = n_fail = 0
    for sid, r in regions.items():
        g = r["grid"]
        if g.max() == 0:
            continue
        name = r["name"].replace("/", "_")
        dest = masks_dir / f"{name}.tiff"
        if dest.exists():
            n_skip += 1
            continue
        # WSI path from slide name: YYYY_MMDDD-suffix -> root/YYYY/MMDDD/name/name.mrxs
        year, month, day = name[:4], name[5:7], name[7:10]
        wsi_p = Path(args.wsi_root) / year / month / day / name / f"{name}.mrxs"
        if not wsi_p.exists():
            n_fail += 1
            print(f"  FAIL {name}: WSI not found at {wsi_p}")
            continue
        # open WSI, get level-0 and level-1 dimensions
        s = openslide.OpenSlide(str(wsi_p))
        l0 = s.level_dimensions[0]
        l1 = s.level_dimensions[1]  # actual level-1 size
        H, W = g.shape
        # scale: level-1 pixel -> level-0 pixel
        sx, sy = l0[0] / l1[0], l0[1] / l1[1]
        mpp_x = float(s.properties["openslide.mpp-x"])
        mpp_y = float(s.properties["openslide.mpp-y"])
        s.close()
        # paint tiles into level-0 array
        arr = np.zeros((l0[1], l0[0]), dtype=np.uint8)
        lut = np.zeros(g.max() + 1, dtype=np.uint8)
        for i in range(1, len(r["regions"]) + 1):
            c = cluster_by_key.get((sid, i))
            if c is not None:
                lut[i] = c + 1
        for row in range(H):
            for col in range(W):
                label = lut[g[row, col]]
                if label == 0:
                    continue
                x0 = int(round((col * 112 + r["x_off"]) * sx))
                y0 = int(round((row * 112 + r["y_off"]) * sy))
                x1 = min(l0[0], x0 + int(round(112 * sx)))
                y1 = min(l0[1], y0 + int(round(112 * sy)))
                if x1 > x0 and y1 > y0:
                    arr[y0:y1, x0:x1] = label
        # write BigTIFF with pyramid
        img = pyvips.Image.new_from_array(arr)
        write_big_tiff(img, dest, mpp_x, mpp_y)
        del arr
        n_masks += 1
        print(f"  [{n_masks}] {dest.name} {l0[1]}x{l0[0]}")
    print(f"  wrote {n_masks}, skipped {n_skip}, failed {n_fail} -> {masks_dir}")

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
    wsi_paths = [wsi_path(r["name"]) for r in regions.values() if r["grid"].max() > 0]
    paths_block = "\n".join(f"    - {p}" for p in sorted(wsi_paths))
    k = args.k
    cfg = f"""# Report-tool config: TileMorph k={k} cluster masks over the MAMMAPRINT WSIs.
#
# This is a Hydra group file for the `report` tool (NOT this repo's Hydra).
# Drop this file as  <config_dir>/reporter/tile_morph_k32.yaml  and run:
#
#     python -m report --config-dir report_conf reporter=tile_morph_k32 user=<you>
#
# (add  mlflow=kubas_external  if you run outside the Kubas cluster)
#
# The mask files are produced by this repo's `make_masks` (one .tiff per slide,
# uint8 label image at WSI level-0 size, pixel value = cluster id + 1
# (the +1 offset), 0 = background).
# The mask file name (slide stem) is the join key to the WSI below.
#
# Adjust the two paths to match where `make_masks` wrote the masks and where
# the WSIs are mounted for xOpat.

defaults:
  - _self_

title: TileMorph clustering (k={k}) - MAMMAPRINT

# --- background: the WSIs to display (MIRAX .mrxs) --------------------------------
# rglob finds every *.mrxs under source_dir; item_name = file stem = mask stem.
background:
  _target_: report.masks.SlideRetriever
  paths:
{paths_block}
  layer_name: MAMMAPRINT WSI (MIRAX)

# --- mask layers: the cluster masks -------------------------------------------------
# Per-cluster on/off: the xOpat colormap shader's `threshold.pips` + `toggleMask`
# render a "Categories" panel with ONE toggle per cluster. Toggle any cluster to
# show/hide it WITHOUT touching the others - that is the per-cluster on/off.
# (This is NOT "32 layers": one label mask per slide already carries all 32
#  clusters; the Categories pips are how xOpat switches individual ones.)
#
# Raw xOpat `colormap` shader (not the built-in `classify`, whose colormap
# assumes class values are uniform over [0,255] and would mis-colour a 0..31
# label image). A raw shader passes `infer_shader_configuration` through
# untouched, so the boundaries we set are exactly what xOpat uses.
#   cluster id c is stored as uint8 value c+1 (the +1 offset); xOpat
#   normalises a pixel to value/255, so the class boundaries sit at c/255
#   (threshold.breaks). Bin 0 (value 0) is background, hidden by
#   threshold.mask[0]=0.
#
# Colors: 33 slots, index i = value i (slot 0 = transparent background,
#   slot i = cluster i-1). Values 1..32 use an even hue-wheel palette with
#   alternating light/saturate so *neighbouring* clusters (the ones that
#   touch on the WSI) are maximally distinct.
mask_retrievers:
  - _target_: report.masks.DriveMasksRetriever
    dir_name: {masks_dir.resolve()}
    layer_name: Clusters (k={k})
    shader_conf:
      type: colormap
      params:
        use_channel0: r
        opacity: 0.55
        connect:
          type: bool
          interactive: false
          default: true
        color:
          type: custom_colormap
          # +1 offset scheme: value i (i=1..32) = cluster i-1; value 0 = background
          default:
          - ''            # value 0 = background -> transparent (masked by threshold.mask[0]=0)
          - "#A64630"      # value 1 = cluster 0
          - "#F4A87B"      # value 2 = cluster 1
          - "#A67230"      # value 3 = cluster 2
          - "#F4D67B"      # value 4 = cluster 3
          - "#A69F30"      # value 5 = cluster 4
          - "#E5F47B"      # value 6 = cluster 5
          - "#81A630"      # value 7 = cluster 6
          - "#B8F47B"      # value 8 = cluster 7
          - "#55A630"      # value 9 = cluster 8
          - "#8AF47B"      # value 10 = cluster 9
          - "#30A638"      # value 11 = cluster 10
          - "#7BF499"      # value 12 = cluster 11
          - "#30A664"      # value 13 = cluster 12
          - "#7BF4C7"      # value 14 = cluster 13
          - "#30A690"      # value 15 = cluster 14
          - "#7BF4F4"      # value 16 = cluster 15
          - "#3090A6"      # value 17 = cluster 16
          - "#7BC7F4"      # value 18 = cluster 17
          - "#3064A6"      # value 19 = cluster 18
          - "#7B99F4"      # value 20 = cluster 19
          - "#3038A6"      # value 21 = cluster 20
          - "#8A7BF4"      # value 22 = cluster 21
          - "#5530A6"      # value 23 = cluster 22
          - "#B87BF4"      # value 24 = cluster 23
          - "#8130A6"      # value 25 = cluster 24
          - "#E5F47B"      # value 26 = cluster 25
          - "#A6309F"      # value 27 = cluster 26
          - "#F47BD6"      # value 28 = cluster 27
          - "#A63072"      # value 29 = cluster 28
          - "#F47BA8"      # value 30 = cluster 29
          - "#A63046"      # value 31 = cluster 30
          - "#4630A6"      # value 32 = cluster 31
          steps: 33
          mode: qualitative
          interactive: true
          title: 'Colormap:'
          continuous: false
        threshold:
          # boundary before class c (c=1..32): value c/255
          breaks:
          - 0.003922
          - 0.007843
          - 0.011765
          - 0.015686
          - 0.019608
          - 0.023529
          - 0.027451
          - 0.031373
          - 0.035294
          - 0.039216
          - 0.043137
          - 0.047059
          - 0.050980
          - 0.054902
          - 0.058824
          - 0.062745
          - 0.066667
          - 0.070588
          - 0.074510
          - 0.078431
          - 0.082353
          - 0.086275
          - 0.090196
          - 0.094118
          - 0.098039
          - 0.101961
          - 0.105882
          - 0.109804
          - 0.113725
          - 0.117647
          - 0.121569
          - 0.125490
          # value 0 = background -> transparent; 1..32 shown
          mask:
          - 0
          - 1
          - 1
          - 1
          - 1
          - 1
          - 1
          - 1
          - 1
          - 1
          - 1
          - 1
          - 1
          - 1
          - 1
          - 1
          - 1
          - 1
          - 1
          - 1
          - 1
          - 1
          - 1
          - 1
          - 1
          - 1
          - 1
          - 1
          - 1
          - 1
          - 1
          - 1
          - 1
          - 1
          interactive: false
          inverted: true
          maskOnly: true
          toggleMask: true
          title: Categories
          min: 0
          max: 1
          minGap: 0.021505
          step: null
          pips:
            mode: count
            values: 33
            density: 4
            stepped: true

# --- where the report is written ----------------------------------------------------
save:
  _target_: report.save.MLFlowReportStorer
  artifact_path: report.html
  metadata:
    experiment_name: Breast Cancer   # MLflow experiment 61 (must exist on the server)
    run_name: TileMorph k{k} cluster masks
    description: Per-slide k={k} cluster masks overlaid on the MAMMAPRINT WSIs
    user: ${{user}}
"""
    dest = out / "report_conf" / "reporter" / "region_clustering.yaml"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(cfg)
    print(f"wrote report config: {dest} ({len(wsi_paths)} slides)")

    print(f"\nmorphology vs majority cluster (slides):")
    print(ct.to_string())
    print(f"wrote: {out} ({time.monotonic()-t0:.0f}s total)")


if __name__ == "__main__":
    main()
