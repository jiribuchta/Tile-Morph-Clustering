# tile_morph_clustering

Unsupervised morphological clustering of the Virchow2 tile embeddings for
MAMAPRINT / MMCI B20-24. Zero-GPU; runs anywhere with `pyarrow + numpy + pandas + scikit-learn + mlflow`.

## Data substrate (verified 2026-09-20)

- Run `e77013b39ac245feacb9d5451c91f646` (exp 61, `Merge Embeddings Virchow2 MMCI B20-24 Train`):
  - `MMCI B20-24 Train_sharded/tiles/*.parquet` — ~9400 parts × ~103 MB, 61,321,694 tiles
  - `MMCI B20-24 Train_sharded/slides/slides.parquet` — 1842 Train-split slides
- Tile schema: `slide_id` (32-byte hex), `x, y` (level-1 px), `tissue_roi_percentage`,
  `blur/folding/residual/epithelium_roi_percentage`, `carcinoma` (0/1), `embedding` (list, dim **2560**)
- Slide schema: `id`, `extent_x/y`, `tile_extent_x/y` (224), `stride_x/y` (112), `mpp_x/y` (~0.47 µm/px),
  `level` (1), `path` (`/mnt/bioptic_tree/...*.mrxs`), `carcinoma` (bool)
- **⚠️ these embeddings failed the 2026-09-20 verification (5/5 slides) and are being regenerated —
  treat everything computed from them as disposable until the new merge run lands.**

## Usage

Hydra module: config `configs/clustering.yaml`, data `configs/data/`,
experiment `configs/experiment/clustering/`. Run from the repo root:

```bash
# full Train set straight from MLflow (downloads all parts — ~1.1 TiB;
# prefer a node where the artifacts are already local, see below)
uv run -m cluster_tiles +data=mmci_b20_24 +experiment/clustering=train_k32

# preview on a few local parts (skip the download)
uv run -m cluster_tiles +data=mmci_b20_24 +experiment/clustering=train_k32 \
  data.paths="/path/to/MMCI B20-24 Train_sharded" parts=24 out=/tmp/preview
```

Any value is a plain CLI override: `k=64 tiles_per_slide=128 min_tissue=0.0
seed=1 out=... parts=...`. Cloud submission: `scripts/run_clustering.py`
(kube `submit_job`).

Knobs that matter:
- `min_tissue=0.0` → include background-ish tiles (you want one "background/glass" cluster to confirm the model sees it)
- `tiles_per_slide=256` → 1842 slides × 256 = 471k vectors ≈ 4.9 GB RAM for X; halve if tight
- `k=32` → tissue-context level; the fine-morphology "pattern" question needs per-cluster sub-clustering or a larger k on a filtered subset, not a bigger k globally

## Outputs (in `--out`)

| file | what |
|---|---|
| `assignments.parquet` | every sampled tile: slide_id, x, y, carcinoma, tissue_roi, cluster, dist_to_centroid |
| `centroids.npy` | k×2560 float32 |
| `X_norm.npy` | the L2-normalized vectors (for re-clustering without re-streaming) |
| `summary.json` | per cluster: n_tiles, n_slides, slide_share, mean_carcinoma, mean_tissue_roi |
| `medoids.jsonl` | 8 nearest-to-centroid tiles per cluster, with crop recipe → montage manifest |
| `slides.parquet` | slide metadata copy |

Montage rendering: for each medoid crop
`reader.read_region((x, y), level=level, width=w, height=h)` (bioforma for `.mrxs`),
grid by cluster, nearest-centroid first.

## Reading results

- `slide_share ≈ 1` + high `mean_tissue_roi` → common tissue archetypes (what you're after)
- `slide_share ≈ 1` + low `mean_tissue_roi` (only with `--min-tissue 0`) → background/fold/blur artifact clusters
- `slide_share ≈ 1` + `mean_carcinoma` skewed → morphology that tracks the carcinoma flag (sanity: MAMAPRINT assay is molecular, so a *strong* link means leak via staining/batch, not biology)
- clusters with `n_slides` ≪ `slide_share×total` → slide-specific batches (scanner/stain) — expect a few; if many, that's the batch effect your unsupervised clusters are *about*
