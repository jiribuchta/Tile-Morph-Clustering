#!/usr/bin/env python3
"""Build a ready-to-run reporter config for the tile_morph_k32 report.

Builds the report from the cluster masks that EXIST in the masks dir right now,
so you can run it on a partial `make_masks` run (or a finished one) without
waiting for the run to complete and write its `manifest.json`.

The two inputs:
  * ``--masks-dir``  dir holding the ``.tiff`` masks (the make_masks output dir).
                     Each ``<stem>.tiff`` is one cluster mask.
  * ``--slides``    the make_masks INPUT ``slides.parquet`` (always present, even
                     mid-run). Maps mask stem -> WSI path, so the report can show
                     each mask on its WSI.

It emits a copy of ``report_conf/reporter/tile_morph_k32.yaml`` with:
  * ``background`` = ``SlideRetriever(paths=<WSIs that have a mask on disk>)``
    (no ``rglob`` over the tree, and only the slides that actually have a mask).
  * mask ``dir_name`` = the masks dir.

Usage:
    python make_report_conf.py --masks-dir /path/to/masks_dir \
                               --slides /path/to/clustering/slides.parquet \
                               [--out /tmp/report_conf]

Then run the report from the generated dir:
    python -m report --config-dir /tmp/report_conf reporter=tile_morph_k32 \\
                     user=YOU mlflow=kubas_external
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd
import yaml

TEMPLATE = Path(__file__).parent / "report_conf" / "reporter" / "tile_morph_k32.yaml"


def _indent_paths(paths: list[str]) -> str:
    """Render the WSI paths as an indented YAML block list under `paths:`."""
    dumped = yaml.safe_dump(paths, default_flow_style=False, sort_keys=False)
    lines = [ln for ln in dumped.splitlines() if ln.strip()]
    return "\n".join("    " + ln if ln else ln for ln in lines)


def wsi_paths_for(masks_dir: Path, slides_path: Path) -> list[str]:
    """WSI paths for exactly the masks that exist on disk, in a stable order."""
    mask_files = sorted(masks_dir.glob("*.tiff"))
    if not mask_files:
        raise SystemExit(f"no *.tiff masks found in {masks_dir}")
    mask_stems = {p.stem for p in mask_files}

    df = pd.read_parquet(slides_path, columns=["path"])
    stem_to_wsi = {Path(str(p)).stem: str(p) for p in df["path"]}

    missing = mask_stems - set(stem_to_wsi)
    if missing:
        raise SystemExit(
            f"{len(missing)} mask stems not in slides.parquet (mismatch?), e.g. "
            + ", ".join(sorted(missing)[:5])
        )

    # stable order: by mask file name (the order make_masks wrote them is not
    # guaranteed); dedupe in case of duplicate stems
    seen: set[str] = set()
    paths: list[str] = []
    for stem in sorted(mask_stems):
        p = stem_to_wsi[stem]
        if p not in seen:
            seen.add(p)
            paths.append(p)
    return paths


def build(masks_dir: Path, slides_path: Path, out_dir: Path) -> Path:
    masks_dir = masks_dir.resolve()
    wsi_paths = wsi_paths_for(masks_dir, slides_path)

    # Gate: every WSI must exist on this machine. A slides.parquet from a
    # different mount would otherwise build a report of slides that show with
    # NO mask layer.
    missing_wsi = [p for p in wsi_paths if not Path(p).exists()]
    if missing_wsi:
        raise SystemExit(
            f"{len(missing_wsi)}/{len(wsi_paths)} WSI paths do not exist on this "
            "machine, e.g.\n  "
            + "\n  ".join(missing_wsi[:5])
            + "\nIs this the mount the report should read from?"
        )

    tpl = TEMPLATE.read_text()

    # background: BasicImageRetriever (rglob) -> SlideRetriever (explicit paths)
    tpl = tpl.replace(
        "_target_: report.masks.BasicImageRetriever",
        "_target_: report.masks.SlideRetriever",
    )
    tpl = re.sub(
        r"source_dir:.*\n\s*globs:.*\n",
        f"paths:\n{_indent_paths(wsi_paths)}\n",
        tpl,
        count=1,
    )
    # mask dir -> the dir that holds the .tiff masks
    tpl = re.sub(r"dir_name:.*\n", f"dir_name: {masks_dir}\n", tpl, count=1)

    # fail fast if any substitution silently missed
    if "report.masks.SlideRetriever" not in tpl or "paths:" not in tpl:
        raise SystemExit("background was not patched to SlideRetriever — check template")
    if f"dir_name: {masks_dir}" not in tpl:
        raise SystemExit("mask dir_name was not patched — check template")

    dest = out_dir / "reporter" / "tile_morph_k32.yaml"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(tpl)

    print(f"wrote {dest}")
    print(f"  WSIs : {len(wsi_paths)}  (the .tiff masks present in {masks_dir})")
    print(f"  masks: {masks_dir}")
    print()
    print("run:")
    print(f"  python -m report --config-dir {out_dir} reporter=tile_morph_k32 \\")
    print("       user=YOU mlflow=kubas_external")
    return dest


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--masks-dir", type=Path, required=True,
                    help="dir holding the .tiff masks (make_masks output dir)")
    ap.add_argument("--slides", type=Path, required=True,
                    help="make_masks INPUT slides.parquet (stem -> WSI path)")
    ap.add_argument("--out", type=Path, default=Path("/tmp/report_conf"),
                    help="dir to write the ready-to-run config into")
    args = ap.parse_args()

    if not args.masks_dir.is_dir():
        raise SystemExit(f"masks dir not found: {args.masks_dir}")
    if not args.slides.is_file():
        raise SystemExit(f"slides.parquet not found: {args.slides}")
    build(args.masks_dir, args.slides, args.out)


if __name__ == "__main__":
    main()
