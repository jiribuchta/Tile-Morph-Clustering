#!/usr/bin/env python3
"""Per-cluster "typical tiles" montages for the k=32 morphology discovery read.

For each cluster, picks the tiles CLOSEST to that cluster's centroid (the
typical members — the morphology a pathologist should expect to see) and
crops them from the WSIs into one 4x4 PNG per cluster. The pathologist reads
the 32 images to decide which clusters are coherent morphologies and which
are mixes.

Phase 1 streams the tile parquets once (the same parts slide_parts.csv points
at — no re-clustering, just similarity to centroids). Phase 2 crops the chosen
tiles from the WSIs at their recorded level/coords (each WSI opened once).

Labels use the +1 offset scheme (cluster c is stored as value c+1 in the
masks), so cluster_<c>.png shows cluster c = mask value c+1.

Usage:
    python make_montage.py \
        --slides    <clustering>/slides.parquet \
        --centroids <clustering>/centroids.npy \
        --parts-csv slide_parts.csv \
        --out /path/to/montage_dir
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pyvips

from make_mask_slide import load_parts_csv
from make_masks import hex_id


def _normalized_embeddings(col, n_rows: int) -> np.ndarray:
    """embedding column -> (n_rows, dim) L2-normalized float32 matrix."""
    emb = col.values.to_numpy(zero_copy_only=False)
    if emb.size % n_rows == 0:
        emb = emb.reshape(n_rows, -1)
    else:
        rows = col.to_numpy(zero_copy_only=False)
        emb = np.stack([np.asarray(r, dtype=np.float32) for r in rows])
    e = emb.astype(np.float32)
    e /= np.linalg.norm(e, axis=1, keepdims=True)
    return e


def select_best(
    part_paths: list[str], centroids: np.ndarray, n: int
) -> dict[int, list[tuple[float, str, int, int]]]:
    """c -> up to n (sim, slide_hex, x, y): the tiles nearest centroid c.

    Per 8192-row batch each cluster only ever considers its top-64 members,
    then keeps the global top-n — Python work is K*64 per batch, not
    K*batch_size.
    """
    Cn = centroids / np.linalg.norm(centroids, axis=1, keepdims=True)
    K = Cn.shape[0]
    best: dict[int, list[tuple[float, str, int, int]]] = {c: [] for c in range(K)}
    for local in part_paths:
        pf = pq.ParquetFile(local)
        for batch in pf.iter_batches(
            columns=["slide_id", "x", "y", "embedding"], batch_size=8192
        ):
            n_rows = len(batch)
            if n_rows == 0:
                continue
            e = _normalized_embeddings(batch.column("embedding"), n_rows)
            sids = batch.column("slide_id").to_numpy(zero_copy_only=False)
            xs = batch.column("x").to_numpy(zero_copy_only=False)
            ys = batch.column("y").to_numpy(zero_copy_only=False)
            sims = e @ Cn.T  # (rows, K) cosine similarity (both normalized)
            for c in range(K):
                s = sims[:, c]
                top = np.argsort(-s)[: min(n_rows, 64)]
                b = best[c]
                b.extend((float(s[i]), hex_id(sids[i]), int(xs[i]), int(ys[i]))
                         for i in top)
                b.sort(reverse=True)
                del b[n:]
    return best


def level_dims(s: pyvips.Image, level: int) -> tuple[int, int]:
    w, h = s.get_width(), s.get_height()
    for _ in range(level):
        w //= 2
        h //= 2
    return w, h


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--slides", type=Path, required=True)
    ap.add_argument("--centroids", type=Path, required=True)
    ap.add_argument("--parts-csv", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n", type=int, default=16, help="tiles per cluster (4x4 grid)")
    ap.add_argument("--size", type=int, default=256,
                    help="display px per tile in the montage")
    args = ap.parse_args()

    slides = pd.read_parquet(args.slides)
    level = int(slides["level"].iloc[0])
    tw = int(slides["tile_extent_x"].iloc[0])
    th = int(slides["tile_extent_y"].iloc[0])
    centroids = np.load(args.centroids).astype(np.float32)
    parts_by_slide = load_parts_csv(args.parts_csv)

    all_parts = sorted({p for ps in parts_by_slide.values() for p in ps})
    slide_path = {hex_id(r.slide_id): str(r.path)
                  for r in slides.itertuples(index=False)}

    print(f"phase 1: selecting top-{args.n} per cluster from {len(all_parts)} parts")
    best = select_best(all_parts, centroids, args.n)
    for c, tiles in best.items():
        print(f"  cluster {c:2d}: {len(tiles)} tiles "
              f"({len({h for _s, h, _x, _y in tiles})} slides)")

    # phase 2: crop the chosen tiles; each WSI opened once
    print("phase 2: cropping from WSIs")
    cells: dict[tuple[int, int, int], pyvips.Image] = {}  # (c, x, y) -> image
    per_slide: dict[str, list[tuple[int, int, int]]] = {}
    for c, tiles in best.items():
        for _sim, hexs, x, y in tiles:
            per_slide.setdefault(hexs, []).append((c, x, y))
    for hexs, items in per_slide.items():
        path = slide_path.get(hexs)
        if path is None:
            print(f"  WARN slide {hexs[:8]} not in slides.parquet")
            continue
        try:
            s = pyvips.Image.new_from_file(path, access="random")
            lref_w, lref_h = level_dims(s, level)
            for c, x, y in items:
                if x >= lref_w or y >= lref_h:
                    continue
                cell = s.crop(x, y, tw, th)
                cells[(c, x, y)] = cell.resize(args.size / tw)
            s.close()
        except Exception as e:  # noqa: BLE001 - one bad WSI can't kill the run
            print(f"  WARN {Path(path).name}: {type(e).__name__}: {e}")

    # compose one 4x4 PNG per cluster
    args.out.mkdir(parents=True, exist_ok=True)
    print(f"phase 3: writing grids -> {args.out}")
    for c, tiles in best.items():
        cells_c = [cells.get((c, x, y)) for _s, _h, x, y in tiles]
        while len(cells_c) < args.n:
            cells_c.append(None)
        rows = []
        for r in range(4):
            row = pyvips.Image.black(args.size * 4, args.size)
            for i in range(4):
                cell = cells_c[r * 4 + i]
                if cell is None:
                    cell = pyvips.Image.black(args.size, args.size)
                elif cell.get_width() != args.size:
                    cell = cell.resize(args.size / cell.get_width())
                row = row.insert(cell, i * args.size, r * args.size)
            rows.append(row)
        grid = pyvips.Image.join(*rows, ncolumns=4)
        dest = args.out / f"cluster_{c:02d}.png"
        grid.pngsave(dest)
        print(f"  {dest.name}: {sum(x is not None for x in cells_c)}/{args.n} tiles")


if __name__ == "__main__":
    main()
