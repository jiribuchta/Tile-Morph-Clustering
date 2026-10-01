#!/usr/bin/env python3
"""Index: which tile-parquet parts hold each slide's tiles.

Scans ONLY the ``slide_id`` column of every tile part (not the embeddings, which
is what made the full run 3.6h) and writes a CSV so you can process one slide at
a time and see which parts each slide needs.

Output: ``<out>.csv`` with columns  slide_id,part_name,part_path
(one row per slide/part pair).  ``slide_id`` is the hex of the raw slide_id
bytes — the same join key ``slides.parquet`` uses after ``hex_id``.

It also prints a summary that answers "is slide-by-slide actually cheaper?":
  * parts-per-slide distribution (min/median/max)
  * how many parts are shared by >1 slide (interleaved) vs exclusive (partitioned)

If parts-per-slide is small and mostly exclusive -> slide-by-slide reads far less
than a full pass (great). If every slide needs (almost) every part -> the data is
interleaved and slide-by-slide multiplies I/O; a single pass that writes each
slide's assignment to disk as it goes is the durable option then.

Usage:
    python slide_parts_index.py --tiles-dir /path/to/tiles [--out slide_parts]
    # or point at a single sharded dir that directly contains *.parquet

The tiles dir is the make_masks/cluster_tiles ``data.paths`` local dir (its
``tiles/`` subfolder, or the folder that directly holds the *.parquet).
"""
from __future__ import annotations

import argparse
import statistics
from pathlib import Path

import pyarrow.parquet as pq


def _hex(v) -> str:
    return v.hex() if isinstance(v, (bytes, bytearray)) else str(v)


def find_tiles_dir(base: Path) -> Path:
    if (base / "tiles").is_dir():
        return base / "tiles"
    return base


def build(tiles_dir: Path, out: Path) -> None:
    parts = sorted(tiles_dir.glob("*.parquet"))
    if not parts:
        raise SystemExit(f"no *.parquet found in {tiles_dir}")

    slide_to_parts: dict[str, list[tuple[str, str]]] = {}
    part_names: list[str] = []
    part_slides: dict[str, set[str]] = {}  # part -> slides (to measure sharing)

    for i, f in enumerate(parts, 1):
        pf = pq.ParquetFile(f)
        part_name = f.name
        part_names.append(part_name)
        slides_in_part: set[str] = set()
        # read ONLY slide_id (cheap); batch to bound memory
        for batch in pf.iter_batches(columns=["slide_id"], batch_size=100_000):
            col = batch.column("slide_id").to_numpy(zero_copy_only=False)
            for v in col:
                slides_in_part.add(_hex(v))
        part_slides[part_name] = slides_in_part
        for s in slides_in_part:
            slide_to_parts.setdefault(s, []).append((part_name, str(f)))
        print(f"  [{i}/{len(parts)}] {part_name}: {len(slides_in_part)} slides")

    # ---- write CSV (long format: slide_id, part_name, part_path) ----
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as fh:
        fh.write("slide_id,part_name,part_path\n")
        for s in sorted(slide_to_parts):
            for part_name, part_path in sorted(slide_to_parts[s]):
                fh.write(f"{s},{part_name},{part_path}\n")

    # ---- summary ----
    per_slide = [len(v) for v in slide_to_parts.values()]
    shared = sum(1 for p in part_slides if len(part_slides[p]) > 1)
    print()
    print(f"wrote {out}")
    print(f"  slides          : {len(slide_to_parts)}")
    print(f"  parts           : {len(parts)}")
    if per_slide:
        print(f"  parts per slide : min={min(per_slide)} "
              f"median={statistics.median(per_slide):.0f} max={max(per_slide)}")
    print(f"  parts holding >1 slide (shared/interleaved): {shared}/{len(parts)}")
    if shared == 0:
        print("  -> parts are exclusive per slide: slide-by-slide is CHEAP "
              "(read only your slide's part).")
    else:
        print("  -> parts are shared across slides: slide-by-slide re-reads shared "
              "parts; a single pass writing per-slide assignments to disk is cheaper.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tiles-dir", type=Path, required=True,
                    help="dir with the tile shards (its tiles/ subdir, or the folder "
                         "that directly holds the *.parquet)")
    ap.add_argument("--out", type=Path, default=Path("slide_parts.csv"))
    args = ap.parse_args()
    tiles_dir = find_tiles_dir(args.tiles_dir)
    if not any(tiles_dir.glob("*.parquet")):
        raise SystemExit(f"no *.parquet under {tiles_dir}")
    build(tiles_dir, args.out)


if __name__ == "__main__":
    main()
