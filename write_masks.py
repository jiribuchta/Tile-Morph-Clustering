"""Write BigTIFF cluster masks + xOpat report config from a completed clustering run.

Loads the clustering results (regions.parquet) from cluster_regions.py, reconstructs
region grids from the tiles CSV, paints cluster labels into WSI-level arrays, and
writes one BigTIFF per slide + the report config.

Usage:
    python write_masks.py \
        --out /path/to/clustering_output \
        --tiles heatmaps_tiles.csv \
        --min-value 0.5 --min-region 8 \
        --k 32 \
        --wsi-root /mnt/bioptic_tree

Requires: regions.parquet in --out (written by cluster_regions.py).
Writes: --out/masks/*.tiff, --out/report_conf/reporter/region_clustering.yaml
"""
import argparse
import csv
import time
from pathlib import Path

import numpy as np
import openslide
import pandas as pd
import pyvips
from ratiopath.masks import write_big_tiff
from scipy import ndimage

ENC = 1_000_000


def load_grids(tiles_csv, min_value, min_region):
    """slide_id -> dict(grid, name, x_off, y_off). Same logic as cluster_regions.load_regions
    but only keeps what's needed for mask painting."""
    per_slide: dict[str, list[tuple[int, int]]] = {}
    names: dict[str, str] = {}
    with open(tiles_csv, newline="") as f:
        for row in csv.DictReader(f):
            v = float(row["value"])
            if v >= min_value:
                per_slide.setdefault(row["slide_id"], []).append(
                    (int(row["x"]), int(row["y"]))
                )
                names.setdefault(row["slide_id"], row["slide_name"])

    out = {}
    for sid, tiles in per_slide.items():
        xs = np.array([t[0] for t in tiles], dtype=np.int32)
        ys = np.array([t[1] for t in tiles], dtype=np.int32)
        cols, rows = xs // 112, ys // 112
        H, W = rows.max() + 1, cols.max() + 1
        x_off = int(xs.min()) % 112
        y_off = int(ys.min()) % 112
        mask = np.zeros((H, W), dtype=bool)
        mask[rows, cols] = True
        lab, n = ndimage.label(mask, structure=np.ones((3, 3), dtype=int))
        counts = np.bincount(lab.ravel(), minlength=n + 1)
        keep = np.flatnonzero(counts[1:] >= min_region) + 1
        remap = np.zeros(n + 1, dtype=np.int32)
        for new, old in enumerate(keep, start=1):
            remap[old] = new
        grid = np.zeros((H, W), dtype=np.int32)
        grid[rows, cols] = remap[lab[rows, cols]]
        out[sid] = {"grid": grid, "name": names[sid], "x_off": x_off, "y_off": y_off}
    return out


def wsi_path(name: str, wsi_root: str) -> Path:
    year, month, day = name[:4], name[5:7], name[7:10]
    return Path(wsi_root) / year / month / day / name / f"{name}.mrxs"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True,
                    help="dir with regions.parquet (from cluster_regions.py)")
    ap.add_argument("--tiles", required=True, help="heatmaps_tiles.csv")
    ap.add_argument("--min-value", type=float, default=0.5)
    ap.add_argument("--min-region", type=int, default=8)
    ap.add_argument("--k", type=int, required=True, help="number of clusters")
    ap.add_argument("--wsi-root", default="/mnt/bioptic_tree")
    ap.add_argument("--slides", default=None,
                    help="only write masks for these slide names (comma-separated)")
    args = ap.parse_args()

    t0 = time.monotonic()
    out = Path(args.out)
    regions_parquet = out / "regions.parquet"
    if not regions_parquet.exists():
        raise SystemExit(f"no {regions_parquet} — run cluster_regions.py first")

    df = pd.read_parquet(regions_parquet)
    cluster_by_key = {
        (row.slide_id, row.region): row.cluster
        for row in df.itertuples(index=False)
    }
    print(f"loaded {len(df)} region assignments from {regions_parquet}")

    print("reconstructing grids from tiles CSV...")
    grids = load_grids(args.tiles, args.min_value, args.min_region)
    print(f"  {len(grids)} slides")

    if args.slides:
        keep = set(args.slides.split(","))
        grids = {k: v for k, v in grids.items() if v["name"] in keep}
        if not grids:
            raise SystemExit(f"none of {args.slides} found")
        print(f"  filtered to {len(grids)} slide(s)")

    masks_dir = out / "masks"
    masks_dir.mkdir(exist_ok=True)
    n_masks = n_skip = n_fail = 0
    for sid, g in grids.items():
        grid = g["grid"]
        if grid.max() == 0:
            continue
        name = g["name"].replace("/", "_")
        dest = masks_dir / f"{name}.tiff"
        if dest.exists():
            n_skip += 1
            continue

        wsi_p = wsi_path(name, args.wsi_root)
        if not wsi_p.exists():
            n_fail += 1
            print(f"  FAIL {name}: WSI not found at {wsi_p}")
            continue

        s = openslide.OpenSlide(str(wsi_p))
        l0 = s.level_dimensions[0]
        l1 = s.level_dimensions[1]
        H, W = grid.shape
        sx, sy = l0[0] / l1[0], l0[1] / l1[1]
        mpp_x = float(s.properties["openslide.mpp-x"])
        mpp_y = float(s.properties["openslide.mpp-y"])
        s.close()

        arr = np.zeros((l0[1], l0[0]), dtype=np.uint8)
        lut = np.zeros(grid.max() + 1, dtype=np.uint8)
        for i in range(1, grid.max() + 1):
            c = cluster_by_key.get((sid, i))
            if c is not None:
                lut[i] = c + 1
        for row in range(H):
            for col in range(W):
                label = lut[grid[row, col]]
                if label == 0:
                    continue
                x0 = int(round((col * 112 + g["x_off"]) * sx))
                y0 = int(round((row * 112 + g["y_off"]) * sy))
                x1 = min(l0[0], x0 + int(round(112 * sx)))
                y1 = min(l0[1], y0 + int(round(112 * sy)))
                if x1 > x0 and y1 > y0:
                    arr[y0:y1, x0:x1] = label

        img = pyvips.Image.new_from_array(arr)
        write_big_tiff(img, dest, mpp_x, mpp_y)
        del arr
        n_masks += 1
        print(f"  [{n_masks}] {dest.name} {l0[1]}x{l0[0]}")

    print(f"  wrote {n_masks}, skipped {n_skip}, failed {n_fail} -> {masks_dir}")

    # report config
    wsi_paths = [str(wsi_path(g["name"], args.wsi_root)) for g in grids.values() if g["grid"].max() > 0]
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
# The mask files are produced by this repo's `write_masks.py` (one .tiff per slide,
# uint8 label image at WSI level-0 size, pixel value = cluster id + 1
# (the +1 offset), 0 = background).
# The mask file name (slide stem) is the join key to the WSI below.
#
# Adjust the two paths to match where `write_masks` wrote the masks and where
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
    print(f"done ({time.monotonic()-t0:.0f}s)")


if __name__ == "__main__":
    main()
