r"""Build per-slide cluster masks (BigTIFF, xOpat-ready) + the xOpat report config.

ONE program controls all of: index -> masks -> report config.

  * index    -- map slide_id -> the tile-parquet parts that hold it. Cheap: reads
               ONLY the ``slide_id`` column (not embeddings) and is CACHED at
               ``out/slide_parts_index.json`` so re-runs resume without re-scanning
               the whole part set. (slide_parts_index.py, folded in)
  * masks    -- one .tiff per slide: uint8 label at WSI level-0 size, pixel value =
               cluster id + 1 (the "+1 offset"), 0 = background. Each tile is
               assigned to its nearest centroid (same KMeans run -> centroids.npy).
               Resumable (a slide whose .tiff exists is skipped) and failure-
               isolated (a WSI that can't be opened is logged + skipped, not fatal).
  * report   -- write a ready-to-run ``reporter/tile_morph_k32.yaml`` from the masks
               that exist on disk + the WSI map. (make_report_conf.py, folded in)

Saved with ratiopath's write_big_tiff (512x512 tiles, DEFLATE, pyramid) at level-0
MPP, so each mask overlays its WSI 1:1 in xOpat.

Tile (x, y) are level-1 top-left coords (tile_extent x/y, level from
slides.parquet); mapped to level 0 with the slide's own level_downsamples.

Run (from repo root, on the cluster where WSIs + tile parquets are mounted):
    uv run -m make_masks +data=mmci_b20_24_train +experiment/masks=mammaprint
    # which slides: slides=5 (first 5) | slides=<slide name/ID> (one slide) | 0 = all
"""

import json
import re
import sys
import time
from pathlib import Path

import hydra
import numpy as np
import openslide
import pandas as pd
import pyarrow.parquet as pq
import pyvips
import yaml
from omegaconf import DictConfig
from rationai.mlkit import autolog
from rationai.mlkit.lightning.loggers import MLFlowLogger
from ratiopath.masks import write_big_tiff
from tqdm import tqdm


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
    pixel under row-major painting) must equal its cluster.
    """
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


# ---- parts index: slide_id -> parts (cheap: slide_id column only, cached) ----

def _embeddings_col(col, n_rows: int) -> np.ndarray:
    """Embedding column -> (n_rows, dim) float32 matrix (uniform fast path)."""
    emb = col.values.to_numpy(zero_copy_only=False)
    if emb.size % n_rows == 0:
        emb = emb.reshape(n_rows, -1)
    else:
        rows = col.to_numpy(zero_copy_only=False)
        emb = np.stack([np.asarray(r, dtype=np.float32) for r in rows])
    return emb.astype(np.float32)


def _slide_ids(batch) -> np.ndarray:
    return batch.column("slide_id").to_numpy(zero_copy_only=False)


def build_index(parts) -> dict[str, list[str]]:
    """One cheap pass over the parts (``slide_id`` column ONLY) -> slide -> parts.

    Records every slide seen so the cache is reusable as the limit grows. Parts
    are interleaved (every slide is in nearly every part), so all parts are
    scanned; the cache is what makes a re-run free.
    """
    index: dict[str, list[str]] = {}
    for _name, local, _size in tqdm(parts, desc="index", unit=" part"):
        pf = pq.ParquetFile(local)
        part_slides: set[str] = set()
        for batch in pf.iter_batches(columns=["slide_id"], batch_size=100_000):
            for v in _slide_ids(batch):
                part_slides.add(hex_id(v))
        for s in part_slides:
            index.setdefault(s, []).append(local)
    return index


def load_index(path: Path) -> dict[str, list[str]] | None:
    if path.is_file():
        return json.loads(path.read_text())
    return None


def save_index(path: Path, index: dict[str, list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(index))


# ---- masks: one .tiff per wanted slide (resumable, failure-isolated) ----

def _assign_batch(batch, target_hex: str, centroids: np.ndarray):
    """(x, y, cluster+1) for the target slide in this batch; nearest-centroid."""
    sids = _slide_ids(batch)
    keep = [i for i, v in enumerate(sids) if hex_id(v) == target_hex]
    if not keep:
        return []
    e = _embeddings_col(batch.column("embedding"), len(batch))[keep].astype(np.float32)
    e /= np.linalg.norm(e, axis=1, keepdims=True)
    clusters = np.argmax(centroids @ e.T, axis=0)
    xs = batch.column("x").to_numpy(zero_copy_only=False)[keep]
    ys = batch.column("y").to_numpy(zero_copy_only=False)[keep]
    return [(int(xs[i]), int(ys[i]), int(clusters[i]) + 1) for i, c in enumerate(clusters)]


def stream_slide_tiles(part_paths: list[str], target_hex: str, centroids: np.ndarray):
    """(x, y, cluster+1) for the target slide, reading ONLY its parts."""
    tiles: list[tuple[int, int, int]] = []
    for local in part_paths:
        pf = pq.ParquetFile(local)
        for batch in pf.iter_batches(columns=["slide_id", "x", "y", "embedding"], batch_size=8192):
            if len(batch) == 0:
                continue
            tiles.extend(_assign_batch(batch, target_hex, centroids))
    return tiles


def process_slides(
    wanted_rows,
    level: int,
    tile_extent: tuple[int, int],
    centroids: np.ndarray,
    index: dict[str, list[str]],
    masks_dir: Path,
) -> tuple[int, int, int, list]:
    """Write one mask per wanted slide; return (written, skipped, failed, rows)."""
    masks_dir.mkdir(parents=True, exist_ok=True)
    ok = skip = fail = 0
    rows = []
    for r in wanted_rows:
        sid = hex_id(r.slide_id)
        dest = masks_dir / f"{Path(str(r.path)).stem}.tiff"
        if dest.exists():
            skip += 1
            print(f"  skip {dest.name}: already written")
            continue
        parts = index.get(sid)
        if not parts:
            fail += 1
            print(f"  FAIL {Path(str(r.path)).name}: no parts in index")
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
        except Exception as e:
            fail += 1
            print(f"  FAIL {Path(str(r.path)).name}: {type(e).__name__}: {e}")
            continue
        ok += 1
        rows.append({"slide_id": sid, "path": str(r.path),
                     "file": f"{Path(str(r.path)).stem}.tiff"})
        print(f"  [{ok}] {dest.name} {arr.shape[1]}x{arr.shape[0]} "
              f"{len(tiles)} tiles ({time.monotonic() - t0:.0f}s)")
    return ok, skip, fail, rows


# ---- report config (make_report_conf, folded in) ----

TEMPLATE = Path(__file__).parent / "report_conf" / "reporter" / "tile_morph_k32.yaml"


def _indent_paths(paths: list[str]) -> str:
    dumped = yaml.safe_dump(paths, default_flow_style=False, sort_keys=False)
    lines = [ln for ln in dumped.splitlines() if ln.strip()]
    return "\n".join("    " + ln if ln else ln for ln in lines)


def wsi_paths_for(masks_dir: Path, slides_path: Path) -> list[str]:
    """WSI paths for exactly the masks present on disk, in a stable order."""
    mask_files = sorted(masks_dir.glob("*.tiff"))
    if not mask_files:
        raise SystemExit(f"no *.tiff masks found in {masks_dir}")
    df = pd.read_parquet(slides_path, columns=["path"])
    stem_to_wsi = {Path(str(p)).stem: str(p) for p in df["path"]}
    seen: set[str] = set()
    paths: list[str] = []
    for p in mask_files:
        w = stem_to_wsi.get(p.stem)
        if w is not None and w not in seen:
            seen.add(w)
            paths.append(w)
    return paths


def write_report_conf(masks_dir: Path, slides_path: Path, out_dir: Path) -> Path | None:
    """Build the xOpat report config from masks on disk + the WSI map.

    Never fatal: if a WSI path isn't present on this machine we warn and skip
    (masks are still valid to copy to the node that has them).
    """
    wsi_paths = wsi_paths_for(masks_dir, slides_path)
    if not wsi_paths:
        print("[report] skipped: no WSI path matches the masks (stem mismatch?)")
        return None
    missing = [p for p in wsi_paths if not Path(p).exists()]
    if missing:
        print(f"[report] skipped: {len(missing)}/{len(wsi_paths)} WSI paths missing on this "
              f"machine (e.g. {missing[0]}) — masks are still valid on the right mount")
        return None
    tpl = TEMPLATE.read_text()
    tpl = tpl.replace("_target_: report.masks.BasicImageRetriever",
                      "_target_: report.masks.SlideRetriever")
    tpl = re.sub(r"source_dir:.*\n\s*globs:.*\n", f"paths:\n{_indent_paths(wsi_paths)}\n", tpl, count=1)
    tpl = re.sub(r"dir_name:.*\n", f"dir_name: {masks_dir.resolve()}\n", tpl, count=1)
    dest = out_dir / "reporter" / "tile_morph_k32.yaml"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(tpl)
    print(f"wrote report config: {dest}  ({len(wsi_paths)} WSIs)")
    return dest


# ---- main ----

def _select_slides(slides: pd.DataFrame, spec) -> pd.DataFrame:
    """`spec`: 0 | "all" = all, int N = first N (carcinoma order), str = a slide name/ID."""
    if spec is None or spec == 0 or (
        isinstance(spec, str) and spec.strip().lower() in {"0", "all"}
    ):
        return slides
    if isinstance(spec, str):
        q = spec.strip().lower()
        m = slides[slides["path"].str.lower().str.contains(q, regex=False)]
        if len(m) == 0:
            raise SystemExit(f"no slide matches {spec!r} in slides.parquet")
        return m
    return slides.head(int(spec))


def main_run(config: DictConfig) -> None:
    sys.stdout.reconfigure(line_buffering=True)
    t0 = time.monotonic()
    out = Path(config.out)
    out.mkdir(parents=True, exist_ok=True)

    clu = Path(config.clustering_out)
    slides = pd.read_parquet(clu / "slides.parquet")
    slides["slide_id"] = slides["slide_id"].apply(hex_id)
    C = np.load(clu / "centroids.npy").astype(np.float32)
    if C.shape[0] >= 255:
        raise SystemExit(f"centroids={C.shape[0]} won't fit the uint8 +1 offset (max 255)")

    level = int(slides["level"].iloc[0])
    tex = (int(slides["tile_extent_x"].iloc[0]), int(slides["tile_extent_y"].iloc[0]))
    slides = _select_slides(slides, config.slides)
    wanted = set(slides["slide_id"])
    print(f"{len(slides)} slide(s), k={C.shape[0]}, out={out}, t0={time.monotonic() - t0:.0f}s")

    # ---- index (cached): slide_id -> parts. This IS the "find the right parquet
    # files" step: once built (or cached), each slide's files are a plain lookup.
    from cluster_tiles import resolve_sources

    parts, _ = resolve_sources(config.data)
    print(f"  {len(parts)} tile parts, ~{sum(p[2] for p in parts) / 1e9:.1f} GB total")
    index_path = out / "slide_parts_index.json"
    index = load_index(index_path)
    if index is not None:
        print(f"  index: loaded {len(index)} slides from {index_path}")
    else:
        print("  index: scanning slide_id (cheap, cached after this run)...")
        ti = time.monotonic()
        index = build_index(parts)
        save_index(index_path, index)
        print(f"  index: {len(index)} slides -> {index_path} ({time.monotonic() - ti:.0f}s)")
    no_parts = [s for s in wanted if s not in index]
    if no_parts:
        print(f"  WARN {len(no_parts)} wanted slide(s) not in index (no tiles), e.g. {no_parts[:3]}")

    # ---- masks: one .tiff per wanted slide (resumable, failure-isolated)
    ok, skip, fail, rows = process_slides(
        list(slides.itertuples(index=False)), level, tex, C, index, out
    )

    (out / "manifest.json").write_text(json.dumps({
        "k": int(C.shape[0]),
        "clustering_out": str(clu),
        "slides": rows,
    }, indent=2))

    # ---- report config (never fatal)
    write_report_conf(out, clu / "slides.parquet", out / "report_conf")

    print(f"wrote {ok} new, {skip} skipped, {fail} failed -> {out} "
          f"({time.monotonic() - t0:.0f}s total)")


def _log_dir(logger: MLFlowLogger, d: Path, artifact_path: str) -> None:
    """Upload a directory's contents (silently skip if it doesn't exist)."""
    if d.exists():
        logger.log_artifacts(str(d), artifact_path)


@hydra.main(config_path="configs", config_name="masks", version_base=None)
@autolog
def main(config: DictConfig, logger: MLFlowLogger) -> None:
    out = Path(config.out)
    main_run(config)
    # manifest (always) + the masks themselves + the xOpat report config, to mlflow
    logger.log_artifacts(str(out / "manifest.json"), "masks")
    _log_dir(logger, out / "masks", "masks")
    _log_dir(logger, out / "report_conf", "report_conf")


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
