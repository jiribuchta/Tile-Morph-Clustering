r"""Build per-slide cluster masks (BigTIFF, xOpat-ready) from clustering outputs.

Writes one .tiff per slide: uint8 label image at the WSI's level-0 size, pixel
value = cluster id, 0 = background. Saved with ratiopath's write_big_tiff
(512x512 tiles, DEFLATE, pyramid) so it overlays the WSI 1:1 in xOpat.

Modes:
  sampled -- paint only the tiles already assigned by cluster_tiles (assignments.parquet).
  full    -- stream the tile parquets, L2-normalize every embedding and assign
             the nearest centroid (same KMeans run -> centroids.npy).

Tile (x, y) are level-1 top-left coords (tile_extent x/y, level from
slides.parquet). Coordinates are mapped to level 0 with the slide's own
level_downsamples, so the mask always matches the WSI geometry.

After each save the mask is re-opened and every painted tile's center pixel is
asserted to equal its cluster (alignment self-check, fails the job on miss).

Run (from repo root, on the cluster where the WSIs are mounted):
    uv run -m make_masks +data=mmci_b20_24 +experiment/masks=mammaprint
"""

import json
import sys
import time
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pyvips
import openslide
from omegaconf import DictConfig
from ratiopath.masks import write_big_tiff
from rationai.mlkit import autolog
from rationai.mlkit.lightning.loggers import MLFlowLogger


def hex_id(v) -> str:
    return v.hex() if isinstance(v, (bytes, bytearray)) else str(v)


def paint(
    arr: np.ndarray,
    x: int,
    y: int,
    w: int,
    h: int,
    label: int,
    scale: tuple[float, float],
) -> None:
    """Paint a tile rect (level-1 top-left) into the level-0 label array."""
    sx, sy = scale
    x0, y0 = int(round(x * sx)), int(round(y * sy))
    x1, y1 = x0 + w, y0 + h
    H, W = arr.shape
    x0c, y0c = max(0, x0), max(0, y0)
    x1c, y1c = min(W, x1), min(H, y1)
    if x1c > x0c and y1c > y0c:
        arr[y0c:y1c, x0c:x1c] = label


def build_slide_mask(
    slide: dict,
    tiles: list[tuple[int, int, int]],
    level: int,
    tile_extent: tuple[int, int],
) -> tuple[np.ndarray, float, float]:
    """tiles: [(x, y, cluster)]; returns (level-0 uint8 array, mpp_x, mpp_y)."""
    s = openslide.OpenSlide(slide["path"])
    try:
        l0 = s.level_dimensions[0]
        lref = s.level_dimensions[level]
        sx, sy = l0[0] / lref[0], l0[1] / lref[1]
        # sanity check only when the slide record carries its own extents
        if "extent_x" in slide and "extent_y" in slide:
            assert np.isclose(lref[0], slide["extent_x"]) and np.isclose(
                lref[1], slide["extent_y"]
            ), f"level-{level} dims {lref} != slides.parquet extent {slide['extent_x']}x{slide['extent_y']}"
        arr = np.zeros((l0[1], l0[0]), dtype=np.uint8)
        tw, th = tile_extent
        # tiles overlap (stride < extent); row-major order makes each tile's
        # top-left corner pixel owned by itself (verify_mask relies on this)
        for x, y, c in sorted(tiles, key=lambda t: (t[1], t[0])):
            paint(arr, x, y, int(round(tw * sx)), int(round(th * sy)), c, (sx, sy))
        # level-0 MPP (mask is level 0); openslide.mpp-x/-y are level-0 values
        mpp_x = float(s.properties["openslide.mpp-x"])
        mpp_y = float(s.properties["openslide.mpp-y"])
        return arr, mpp_x, mpp_y
    finally:
        s.close()


def verify_mask(path: Path, tiles: list[tuple[int, int, int]], level: int) -> None:
    """Re-open the written mask; each tile's top-left corner pixel (its exclusive
    pixel under row-major painting) must equal its cluster."""
    s = openslide.OpenSlide(str(path))
    try:
        l0 = s.level_dimensions[0]
        lref = s.level_dimensions[level]
        sx, sy = l0[0] / lref[0], l0[1] / lref[1]
        bad = 0
        for x, y, c in tiles:
            px = int(round(x * sx)) + 1
            py = int(round(y * sy)) + 1
            px, py = min(px, l0[0] - 1), min(py, l0[1] - 1)
            px, py = max(0, px), max(0, py)
            v = s.read_region((px, py), 0, (1, 1)).getpixel((0, 0))[0]
            if v != c:
                bad += 1
        if bad:
            raise AssertionError(f"{bad}/{len(tiles)} tiles misaligned in {path.name}")
    finally:
        s.close()


def stream_tiles(parts, slide_hexes: set[str], centroids: np.ndarray) -> dict[str, list[tuple[int, int, int]]]:
    """Single pass over tile parquets; paint nearest-centroid cluster for the wanted slides."""
    acc: dict[str, list[tuple[int, int, int]]] = {h: [] for h in slide_hexes}
    C = centroids
    for _name, local, _size in parts:
        pf = pq.ParquetFile(local)
        for batch in pf.iter_batches(columns=["slide_id", "x", "y", "embedding"], batch_size=8192):
            n = len(batch)
            if n == 0:
                continue
            sids = batch.column("slide_id").to_numpy(zero_copy_only=False)
            keep = [i for i, v in enumerate(sids) if hex_id(v) in slide_hexes]
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
            clusters = np.argmax(C @ e.T, axis=1)
            for i, k in enumerate(keep):
                acc[hex_id(sids[k])].append(
                    (int(xs[i]), int(ys[i]), int(clusters[i]))
                )
    return acc


def main_run(config: DictConfig) -> None:
    sys.stdout.reconfigure(line_buffering=True)
    t0 = time.monotonic()
    out = Path(config.out)
    out.mkdir(parents=True, exist_ok=True)

    clu = Path(config.clustering_out)
    slides = pd.read_parquet(clu / "slides.parquet")
    slides["slide_id"] = slides["slide_id"].apply(hex_id)
    assignments = pd.read_parquet(clu / "assignments.parquet")
    assignments["slide_id"] = assignments["slide_id"].apply(hex_id)
    centroids = np.load(clu / "centroids.npy").astype(np.float32)

    if int(config.slides) > 0:
        slides = slides.head(int(config.slides))
    wanted = set(slides["slide_id"])
    print(f"{len(slides)} slides, mode={config.mode}, out={out}")

    # resolve tile parquet parts (only needed for full mode)
    parts = []
    if config.mode == "full":
        from cluster_tiles import resolve_sources

        parts, _ = resolve_sources(config.data)
        print(f"  {len(parts)} tile parts, ~{sum(p[2] for p in parts) / 1e9:.1f} GB")

    level = int(slides["level"].iloc[0])
    tex = (int(slides["tile_extent_x"].iloc[0]), int(slides["tile_extent_y"].iloc[0]))

    tiles_by_slide: dict[str, list[tuple[int, int, int]]]
    if config.mode == "sampled":
        sub = assignments[assignments["slide_id"].isin(wanted)]
        tiles_by_slide = {
            h: list(zip(g["x"], g["y"], g["cluster"].astype(int)))
            for h, g in sub.groupby("slide_id", sort=False)
        }
    else:
        t1 = time.monotonic()
        tiles_by_slide = stream_tiles(parts, wanted, centroids)
        print(f"  streamed embeddings for {len(wanted)} slides ({time.monotonic() - t1:.0f}s)")

    done = 0
    for _, row in slides.iterrows():
        h = row["slide_id"]
        tiles = tiles_by_slide.get(h, [])
        if not tiles:
            print(f"  skip {row['path'].rsplit('/', 1)[-1]}: no tiles")
            continue
        arr, mpp_x, mpp_y = build_slide_mask(
            row.to_dict(), tiles, level, tex
        )
        dest = out / f"{Path(row['path']).stem}.tiff"
        img = pyvips.Image.new_from_array(arr)
        write_big_tiff(img, dest, mpp_x, mpp_y)
        verify_mask(dest, tiles, level)
        done += 1
        print(
            f"  [{done}/{len(slides)}] {dest.name} {arr.shape[1]}x{arr.shape[0]} "
            f"{len(tiles)} tiles ({time.monotonic() - t0:.0f}s)"
        )

    (out / "manifest.json").write_text(json.dumps({
        "mode": config.mode,
        "k": int(centroids.shape[0]),
        "clustering_out": str(clu),
        "slides": [
            {"slide_id": r["slide_id"], "path": r["path"], "file": f"{Path(r['path']).stem}.tiff"}
            for _, r in slides.iterrows()
        ],
    }, indent=2))
    print(f"wrote {done} masks to {out} ({time.monotonic() - t0:.0f}s total)")


@hydra.main(config_path="configs", config_name="masks", version_base=None)
@autolog
def main(config: DictConfig, logger: MLFlowLogger) -> None:
    main_run(config)
    logger.log_artifacts(str(Path(config.out) / "manifest.json"), "masks")


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
