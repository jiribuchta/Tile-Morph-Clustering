"""Render cluster medoid tiles into a labeled contact sheet (``medoids.png``).

One shared module for two entry points:
- ``cluster_tiles.py`` renders it at the end of a clustering job (needs only the job env),
- this file's ``main`` re-renders from an existing ``medoids.jsonl`` without rerunning
  the clustering (needs a node with the WSIs mounted).

Layout: rows = clusters (ascending id), columns = medoid tiles, distance to
centroid printed per tile, row label ``cluster <id>``.

Run standalone:
    uv run python scripts/montage.py --medoids path/to/medoids.jsonl --out .
"""

import argparse
import json
from pathlib import Path

import openslide
from PIL import Image, ImageDraw


def crop_tile(
    slide_path: str, x: int, y: int, level: int, w: int, h: int, size: int
) -> Image.Image:
    with openslide.OpenSlide(slide_path) as slide:
        img = slide.read_region((x, y), level, (w, h))
    return img.resize((size, size))


def _cluster_order(by_cluster: dict[int, list[dict]], k: int, sort: str) -> list[int]:
    if sort == "tissue":
        means = {
            c: sum(r["tissue"] for r in rs) / len(rs)
            for c, rs in by_cluster.items()
            if rs and all("tissue" in r for r in rs)
        }
        if len(means) == len(by_cluster) and means:
            return sorted(range(k), key=lambda c: -means.get(c, -1.0))
    return list(range(k))


def render_medoids(
    rows: list[dict], out_dir: Path, cols: int = 8, size: int = 256, sort: str = "id"
) -> Path:
    """Write ``medoids.png`` into ``out_dir``; returns the path.

    ``rows``: medoid dicts with keys cluster, slide_path, x, y, level, w, h, dist.
    """
    by_cluster: dict[int, list[dict]] = {}
    for r in rows:
        by_cluster.setdefault(int(r["cluster"]), []).append(r)
    k = max(int(r["cluster"]) for r in rows) + 1

    label_h, gap = 22, 2
    cell = size
    width = cols * cell + (cols + 1) * gap
    height = k * (label_h + cell + gap) + gap
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    order = _cluster_order(by_cluster, k, sort)
    for i, c in enumerate(order):
        y0 = gap + i * (label_h + cell + gap)
        draw.text((gap, y0), f"cluster {c}", fill="black")
        y1 = y0 + label_h
        for j, r in enumerate(by_cluster.get(c, [])[:cols]):
            x1 = gap + j * (cell + gap)
            tile = crop_tile(
                r["slide_path"],
                int(r["x"]),
                int(r["y"]),
                int(r["level"]),
                int(r["w"]),
                int(r["h"]),
                size,
            )
            canvas.paste(tile, (x1, y1))
            d = r.get("dist")
            if d is not None:
                draw.rectangle([x1, y1, x1 + 34, y1 + 14], fill="white")
                draw.text((x1 + 2, y1 + 2), f"{d:.2f}", fill="black")

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "medoids.png"
    canvas.save(out_path)
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--medoids", required=True, help="path to medoids.jsonl from the clustering job"
    )
    ap.add_argument("--out", default=".", help="output dir for medoids.png")
    ap.add_argument("--cols", type=int, default=8, help="medoids per cluster row")
    ap.add_argument("--size", type=int, default=256, help="render tile size (px)")
    ap.add_argument(
        "--sort",
        choices=["id", "tissue"],
        default="id",
        help="row order: cluster id, or densest-tissue first (needs tissue in medoids.jsonl)",
    )
    ap.add_argument(
        "--path-map",
        nargs="*",
        default=[],
        metavar="OLD_PREFIX=NEW_PREFIX",
        help="remap slide_path prefixes, e.g. /mnt/bioptic_tree=/mnt/data/... (repeatable)",
    )
    args = ap.parse_args()

    with open(args.medoids) as f:
        rows = [json.loads(line) for line in f if line.strip()]
    for pair in args.path_map:
        old, new = pair.split("=", 1)
        for r in rows:
            if r["slide_path"].startswith(old):
                r["slide_path"] = new + r["slide_path"][len(old) :]
    out_path = render_medoids(
        rows, Path(args.out), cols=args.cols, size=args.size, sort=args.sort
    )
    print(f"wrote: {out_path} ({len(rows)} tiles)")


if __name__ == "__main__":
    main()
