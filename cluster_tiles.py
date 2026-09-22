r"""Unsupervised morphological clustering of Virchow2 tile embeddings (MAMAPRINT / MMCI B20-24).

Streams tile-embedding parquet parts (mlflow artifacts or a local dir), subsamples
tiles per slide (deterministic stride in tiling order -> spatially spread),
L2-normalizes, runs KMeans, and writes to `out`:
  slides.parquet, X_norm.npy, centroids.npy, assignments.parquet, summary.json, medoids.jsonl

Tile parquet schema (verified on run e77013b39ac245feacb9d5451c91f646):
  slide_id: binary[32] (hex id, joins slides.parquet `id`)
  x, y: int64 tile top-left, level-1 space (slides.parquet level=1, mpp~0.47 um/px)
  tissue_roi_percentage, blur_percentage, folding_percentage, residual_percentage,
  epithelium_roi_percentage: float64
  carcinoma: int64 0/1 (tile-level classifier)
  embedding: list<float64>  (dim 2560, Virchow2)

Montage: crop a WSI with reader.read_region((x, y), level=slide_level, w=tile_extent_x, h=tile_extent_y).

Run (from repo root):
    uv run -m cluster_tiles +data=mmci_b20_24 +experiment/clustering=train_k32
Preview on a few local parts:
    uv run -m cluster_tiles +data=mmci_b20_24 +experiment/clustering=train_k32 \
        data.paths="/path/to/MMCI B20-24 Train_sharded" parts=24

Data server comes from MLFLOW_TRACKING_URI (see MLFLOW.md).
"""

import json
import os
import sys
import time
from pathlib import Path

import hydra
import mlflow
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from omegaconf import DictConfig
from rationai.mlkit import autolog
from rationai.mlkit.lightning.loggers import MLFlowLogger


CHEAP_COLUMNS = ["slide_id", "x", "y", "carcinoma", "tissue_roi_percentage"]


def resolve_sources(data: DictConfig) -> tuple[list[tuple[str, str, int]], str]:
    """Return list of (name, local_path, size) for tile parts; also local slides.parquet path.

    ``data.paths`` (a local sharded folder, or ``{local: ...}``) wins when set;
    otherwise the ``data.mlflow_uris`` artifacts are downloaded (server = MLFLOW_TRACKING_URI).
    """
    paths = data.get("paths", None)
    local_dir = (
        str(paths)
        if isinstance(paths, str)
        else (paths.local if paths is not None else None)
    )
    if local_dir is not None:
        d = Path(local_dir)
        tiles = d / "tiles" if (d / "tiles").is_dir() else d
        files = sorted(tiles.glob("*.parquet"))
        parts = [(f.name, str(f), f.stat().st_size) for f in files]
        slides = d / "slides" / "slides.parquet"
        if not slides.exists():
            slides = d / "slides.parquet"
        return parts, str(slides)

    tiles_uri, slides_uri = data.mlflow_uris.tiles, data.mlflow_uris.slides
    parts = []
    for f in mlflow.artifacts.list_artifacts(tiles_uri):
        if f.is_dir or not f.path_name.endswith(".parquet"):
            continue
        local = mlflow.artifacts.download_artifacts(f"{tiles_uri}/{f.path_name}")
        if os.path.isdir(local):
            local = os.path.join(local, min(os.listdir(local)))
        parts.append((f.path_name, local, f.file_size))
    slides_dl = mlflow.artifacts.download_artifacts(slides_uri)
    if os.path.isdir(slides_dl):
        slides_dl = os.path.join(slides_dl, min(os.listdir(slides_dl)))
    return parts, slides_dl


def stream_rows(parts, with_embedding):
    """Yield (part_name, df, emb|None) per batch. Row position in df = row index.

    ``df`` holds CHEAP_COLUMNS; when embedding is requested, ``emb`` is a float32
    (n, dim) array taken straight from arrow (no per-cell Python objects).
    """
    for _name, local, _size in parts:
        pf = pq.ParquetFile(local)
        cols = CHEAP_COLUMNS + (["embedding"] if with_embedding else [])
        for batch in pf.iter_batches(columns=cols, batch_size=4096):
            n = len(batch)
            if n == 0:
                continue
            df = batch.select(CHEAP_COLUMNS).to_pandas()
            emb = None
            if with_embedding:
                vals = batch.column("embedding").values.to_numpy(zero_copy_only=False)
                if vals.size % n == 0:  # uniform embedding dim -> fast path
                    emb = vals.reshape(n, -1).astype(np.float32)
                else:  # fallback: ragged rows
                    rows = batch.column("embedding").to_numpy(zero_copy_only=False)
                    emb = np.stack([np.asarray(r, dtype=np.float32) for r in rows])
            yield _name, df, emb


def run_clustering(config: DictConfig, logger: MLFlowLogger) -> None:
    sys.stdout.reconfigure(
        line_buffering=True
    )  # job logs are not a tty; keep prints live
    t0 = time.monotonic()
    out = Path(config.out)
    out.mkdir(parents=True, exist_ok=True)

    print("resolving sources...")
    parts, slides_path = resolve_sources(config.data)
    if not parts:
        raise SystemExit("no tile parts found")
    parts = parts[: config.parts] if config.parts else parts
    print(f"  {len(parts)} tile parts, ~{sum(p[2] for p in parts) / 1e9:.1f} GB total")

    slides = pd.read_parquet(slides_path)
    slides["slide_id"] = slides["id"].apply(
        lambda b: b.hex() if isinstance(b, (bytes, bytearray)) else str(b)
    )
    slides.to_parquet(out / "slides.parquet")
    print(
        f"  {len(slides)} slides, level={slides['level'].unique().tolist()}, "
        f"mpp~{slides['mpp_x'].mean():.3f} um/px, carcinoma slides={int(slides['carcinoma'].sum())}"
    )
    slide_meta = slides.set_index("slide_id")[
        ["path", "level", "tile_extent_x", "tile_extent_y", "mpp_x", "carcinoma"]
    ]

    # ---- pass 1: count eligible rows per slide (cheap columns only)
    print("pass 1: counting eligible tiles per slide...")
    eligible = {}  # slide_id(hex) -> n
    cur = None
    for _name, df, _ in stream_rows(parts, with_embedding=False):
        if _name != cur:
            cur = _name
            print(f"  pass1 {_name} ({time.monotonic() - t0:.0f}s)")
        m = df["tissue_roi_percentage"].to_numpy() >= config.min_tissue
        for sid, ok in zip(df["slide_id"], m, strict=True):
            if ok:
                h = sid.hex() if isinstance(sid, (bytes, bytearray)) else str(sid)
                eligible[h] = eligible.get(h, 0) + 1
    print(
        f"  {len(eligible)} slides with eligible tiles, "
        f"{sum(eligible.values())} eligible rows total"
    )

    # ---- decide per-slide plan: stride sampling in tiling order
    rng = np.random.default_rng(config.seed)
    plan = {}  # slide_hex -> [step, offset, cap, kept, rank]
    for sid, n in eligible.items():
        cap = min(config.tiles_per_slide, n)
        step = max(1, n // cap)
        offset = int(rng.integers(0, step))
        plan[sid] = [step, offset, cap, 0, 0]

    # ---- pass 2: keep rows (stride), read embeddings only here
    print("pass 2: collecting embeddings...")
    X_rows, meta_rows = [], []
    kept, cur = 0, None
    for _name, df, emb in stream_rows(parts, with_embedding=True):
        if _name != cur:
            if cur is not None:
                print(f"  pass2 {cur} done, kept {kept} ({time.monotonic() - t0:.0f}s)")
            cur = _name
            print(f"  pass2 {_name} ({time.monotonic() - t0:.0f}s)")
        m = df["tissue_roi_percentage"].to_numpy() >= config.min_tissue
        if not m.any():
            continue
        sid_hex = [
            s.hex() if isinstance(s, (bytes, bytearray)) else str(s)
            for s in df["slide_id"]
        ]
        keep_idx = []
        for i, ok in enumerate(m):
            if not ok:
                continue
            h = sid_hex[i]
            p = plan.get(h)
            if p is None:
                continue
            # stride sample in tiling order: keep slot (rank+offset) % step == 0, stop at cap
            if p[3] < p[2] and (p[4] + p[1]) % p[0] == 0:
                keep_idx.append(i)
                kept += 1
                p[3] += 1
            p[4] += 1
        if keep_idx:
            X_rows.append(emb[keep_idx].copy())
            ki = np.asarray(keep_idx)
            meta_rows.extend(
                zip(
                    [sid_hex[i] for i in keep_idx],
                    df["x"].to_numpy()[ki].astype(int),
                    df["y"].to_numpy()[ki].astype(int),
                    df["carcinoma"].to_numpy()[ki].astype(int),
                    df["tissue_roi_percentage"].to_numpy()[ki].astype(float),
                    strict=True,
                )
            )
    if cur is not None:
        print(f"  pass2 {cur} done, kept {kept} ({time.monotonic() - t0:.0f}s)")
    print(f"  collected {kept} tile embeddings ({time.monotonic() - t0:.0f}s)")

    if not X_rows:
        raise SystemExit("no tiles collected")
    X = np.concatenate(X_rows, axis=0)  # f32 in, f32 out - no f64 copy
    del X_rows
    dim = X.shape[1]

    # L2 normalize (in place)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    X /= norms
    np.save(out / "X_norm.npy", X)

    # ---- cluster
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    print(
        f"fitting KMeans k={config.k} n_init={config.n_init} on {X.shape} "
        f"({time.monotonic() - t0:.0f}s in)"
    )
    km = KMeans(
        n_clusters=config.k, n_init=config.n_init, random_state=config.seed
    ).fit(X)
    labels = km.labels_
    n_sil = min(5000, len(labels))
    idx_sil = rng.choice(len(labels), n_sil, replace=False)
    sil = (
        silhouette_score(X[idx_sil], labels[idx_sil])
        if n_sil > config.k
        else float("nan")
    )
    print(f"  inertia={km.inertia_:.1f}  silhouette(subsample)={sil:.3f}")

    dist = np.linalg.norm(X - km.cluster_centers_[labels], axis=1)
    df_out = pd.DataFrame(
        {
            "slide_id": [r[0] for r in meta_rows],
            "x": [r[1] for r in meta_rows],
            "y": [r[2] for r in meta_rows],
            "carcinoma": [r[3] for r in meta_rows],
            "tissue_roi": [r[4] for r in meta_rows],
            "cluster": labels,
            "dist_to_centroid": dist.astype(np.float32),
        }
    )
    df_out.to_parquet(out / "assignments.parquet")
    np.save(out / "centroids.npy", km.cluster_centers_.astype(np.float32))

    # ---- summary + medoids (montage manifest)
    summary = {}
    medoids = []
    for c in range(config.k):
        m = df_out["cluster"] == c
        sids = df_out.loc[m, "slide_id"].unique()
        summary[str(c)] = {
            "n_tiles": int(m.sum()),
            "n_slides": len(sids),
            "slide_share": round(len(sids) / max(1, df_out["slide_id"].nunique()), 4),
            "mean_carcinoma": round(float(df_out.loc[m, "carcinoma"].mean()), 3),
            "mean_tissue_roi": round(float(df_out.loc[m, "tissue_roi"].mean()), 4),
        }
        cm = np.flatnonzero(labels == c)
        if len(cm) == 0:
            continue
        d = np.linalg.norm(X[cm] - km.cluster_centers_[c], axis=1)
        for j in cm[np.argsort(d)[:8]]:
            h = meta_rows[j][0]
            sm = slide_meta.loc[h]
            medoids.append(
                {
                    "cluster": int(c),
                    "slide_id": h,
                    "slide_path": sm["path"],
                    "x": int(meta_rows[j][1]),
                    "y": int(meta_rows[j][2]),
                    "level": int(sm["level"]),
                    "w": int(sm["tile_extent_x"]),
                    "h": int(sm["tile_extent_y"]),
                    "mpp": float(sm["mpp_x"]),
                    "dist": round(float(d[np.where(cm == j)[0][0]]), 4),
                }
            )
    (out / "summary.json").write_text(
        json.dumps(
            {
                "k": config.k,
                "n_tiles": len(meta_rows),
                "n_slides": df_out["slide_id"].nunique(),
                "dim": dim,
                "silhouette_subsample": round(float(sil), 4),
                "tiles_per_slide": config.tiles_per_slide,
                "min_tissue": config.min_tissue,
                "seed": config.seed,
                "part_limit": config.parts,
                "clusters": summary,
            },
            indent=2,
        )
    )
    with (out / "medoids.jsonl").open("w") as f:
        for r in medoids:
            f.write(json.dumps(r) + "\n")

    # ---- log to mlflow
    logger.log_artifacts(str(out), config.mlflow_artifact_path)
    mlflow.log_metrics(
        {
            "k": float(config.k),
            "n_tiles": float(len(meta_rows)),
            "n_slides": float(df_out["slide_id"].nunique()),
            "silhouette_subsample": float(sil),
            "inertia": float(km.inertia_),
        }
    )

    print(f"wrote: {out} ({time.monotonic() - t0:.0f}s total)")
    print("clusters by n_tiles (top 10):")
    top = sorted(summary.items(), key=lambda kv: -kv[1]["n_tiles"])[:10]
    for c, s in top:
        print(
            f"  {c:>3}: {s['n_tiles']:>6} tiles, {s['n_slides']:>4} slides, "
            f"carcin={s['mean_carcinoma']:.2f}, tissue={s['mean_tissue_roi']:.3f}"
        )


@hydra.main(config_path="configs", config_name="clustering", version_base=None)
@autolog
def main(config: DictConfig, logger: MLFlowLogger) -> None:
    run_clustering(config, logger)


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
