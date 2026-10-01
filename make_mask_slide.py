#!/usr/bin/env python3
"""Build one slide's cluster mask — durable, resumable, failure-isolated.

Reuses ``make_masks``' ``build_slide_mask`` / ``verify_mask`` / ``hex_id``. For a
given slide it:
  * reads ONLY the tile parts that hold that slide's tiles (from ``slide_parts.csv``),
  * assigns each tile its nearest-centroid cluster,
  * opens the WSI and writes ``<stem>.tiff`` into the masks dir.

Resumable: a slide whose ``.tiff`` already exists is skipped. Failure-isolated:
a WSI that fails to open (e.g. not mounted) is logged and skipped, NOT fatal — so
one bad slide can't kill the run and the others still get their masks.

Usage:
    python make_mask_slide.py \
        --slides  <clustering>/slides.parquet \
        --centroids <clustering>/centroids.npy \
        --parts-csv slide_parts.csv \
        --masks-dir /path/to/masks_dir \
        --all                       # or: --slide <wsi stem or slide_id>
"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pyvips

from make_masks import build_slide_mask, verify_mask, hex_id
from ratiopath.masks import write_big_tiff


def load_parts_csv(path: Path) -> dict[str, list[str]]:
    """slide_id(hex) -> [part paths]."""
    m: dict[str, list[str]] = {}
    with path.open() as fh:
        for r in csv.DictReader(fh):
            m.setdefault(r["slide_id"], []).append(r["part_path"])
    return m


def stream_slide_tiles(
    part_paths: list[str], target_hex: str, centroids: np.ndarray
) -> list[tuple[int, int, int]]:
    """(x, y, cluster) for the target slide, reading only its parts."""
    C = centroids
    tiles: list[tuple[int, int, int]] = []
    for local in part_paths:
        pf = pq.ParquetFile(local)
        for batch in pf.iter_batches(
            columns=["slide_id", "x", "y", "embedding"], batch_size=8192
        ):
            n = len(batch)
            if n == 0:
                continue
            sids = batch.column("slide_id").to_numpy(zero_copy_only=False)
            keep = [i for i, v in enumerate(sids) if hex_id(v) == target_hex]
            if not keep:
                continue
            emb = batch.column("embedding").values.to_numpy(zero_copy_only=False)
            if emb.size % n == 0:
                emb = emb.reshape(n, -1)
            else:
                rows = batch.column("embedding").to_numpy(zero_copy_only=False)
                emb = np.stack([np.asarray(r, dtype=np.float32) for r in rows])
            xs = batch.column("x").to_numpy(zero_copy_only=False)[keep]
            ys = batch.column("y").to_numpy(zero_copy_only=False)[keep]
            e = emb[keep].astype(np.float32)
            norms = np.linalg.norm(e, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            e /= norms
            clusters = np.argmax(C @ e.T, axis=0)
            for i, k in enumerate(keep):
                tiles.append((int(xs[i]), int(ys[i]), int(clusters[i])))
    return tiles


def _targets(spec: str | None, all_: bool, slides: pd.DataFrame):
    """Return the list of slide rows to process."""
    if all_:
        return list(slides.itertuples(index=False))
    if spec:
        rows = [
            r for r in slides.itertuples(index=False)
            if hex_id(r.slide_id) == spec or Path(str(r.path)).stem == spec
        ]
        if not rows:
            raise SystemExit(f"slide {spec!r} not in slides.parquet")
        return rows
    raise SystemExit("give --slide <stem|id> or --all")


def process(
    slides: pd.DataFrame,
    level: int,
    tile_extent: tuple[int, int],
    centroids: np.ndarray,
    parts_by_slide: dict[str, list[str]],
    masks_dir: Path,
    targets: list,
) -> tuple[int, int, int]:
    """Write one mask per target slide; return (written, skipped, failed)."""
    masks_dir.mkdir(parents=True, exist_ok=True)
    ok = skip = fail = 0
    for r in targets:
        sid = hex_id(r.slide_id)
        dest = masks_dir / f"{Path(str(r.path)).stem}.tiff"
        if dest.exists():
            skip += 1
            print(f"  skip {dest.name}: already written")
            continue
        parts = parts_by_slide.get(sid)
        if not parts:
            fail += 1
            print(f"  FAIL {Path(str(r.path)).name}: no parts in csv")
            continue
        t0 = time.monotonic()
        try:
            tiles = stream_slide_tiles(parts, sid, centroids)
            if not tiles:
                fail += 1
                print(f"  FAIL {Path(str(r.path)).name}: no tiles for {sid[:8]}")
                continue
            arr, mpp_x, mpp_y = build_slide_mask(
                {"path": str(r.path)}, tiles, level, tile_extent
            )
            img = pyvips.Image.new_from_array(arr)
            write_big_tiff(img, dest, mpp_x, mpp_y)
            verify_mask(dest, tiles, level)
        except Exception as e:  # noqa: BLE001 - isolate per-slide failures
            fail += 1
            print(f"  FAIL {Path(str(r.path)).name}: {type(e).__name__}: {e}")
            continue
        ok += 1
        print(f"  [{ok}] {dest.name} {arr.shape[1]}x{arr.shape[0]} "
              f"{len(tiles)} tiles ({time.monotonic() - t0:.0f}s)")
    return ok, skip, fail


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--slides", type=Path, required=True)
    ap.add_argument("--centroids", type=Path, required=True)
    ap.add_argument("--parts-csv", type=Path, required=True)
    ap.add_argument("--masks-dir", type=Path, required=True)
    ap.add_argument("--slide", help="one WSI stem or slide_id(hex)")
    ap.add_argument("--all", action="store_true",
                    help="process every slide in slides.parquet (skip done)")
    args = ap.parse_args()

    slides = pd.read_parquet(args.slides)
    level = int(slides["level"].iloc[0])
    tex = (int(slides["tile_extent_x"].iloc[0]), int(slides["tile_extent_y"].iloc[0]))
    centroids = np.load(args.centroids).astype(np.float32)
    parts_by_slide = load_parts_csv(args.parts_csv)

    targets = _targets(args.slide, args.all, slides)
    ok, skip, fail = process(slides, level, tex, centroids, parts_by_slide,
                             args.masks_dir, targets)
    print(f"\ndone: {ok} written, {skip} skipped (already), {fail} failed")


if __name__ == "__main__":
    main()
