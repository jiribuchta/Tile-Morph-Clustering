# AGENTS.md — Tile-Morph-Clustering

Repo of tile-level morphology analysis for breast-cancer WSIs (MAMMAPRINT project).
Everything here answers one question: **can tile embeddings (Virchow2) separate
ductal (ICD-O 8500) from lobular (8520) carcinoma morphology?**

## Environment & conventions

- Python via `uv` (`uv sync`, run scripts with `uv run python ...`).
- Ruff + mypy configured (`.ruff.toml`, `.mypy.ini`).
- `ml/`, `configs/`, `preprocessing/` are leftovers of the ML template repo
  (PyTorch Lightning + Hydra + MLflow). The morphology pipeline does **not**
  use them — only `mlflow` artifact download in `embeddings_to_heatmap.py`.
- Jobs run on the CERIT Kubernetes cluster. Submission wrappers live in
  `scripts/run_*.py` (see "Cluster jobs" below). Jobs clone this repo on the
  pod and `uv sync`, so **changes must be pushed before submitting**.

## Data locations (cluster mounts)

| Path | Contents |
|---|---|
| `/mnt/projects/breast_cancer/bc/tile_embeddings/virchow2/MMCI B20-24 Train` / `... Val` | per-slide `.pt` files with Virchow2 tile embeddings (2560-d, L2-normalized) |
| `/mnt/projects/breast_cancer/bc/heatmaps/...` | per-slide parquets from `embeddings_to_heatmap.py` |
| `/mnt/bioptic_tree/YYYY/MM/DD/<slide>/<slide>.mrxs` | whole-slide images (bioptic tree layout, slide name encodes the date) |
| `/mnt/projects/breast_cancer/tile_morph_clustering/` | all pipeline outputs (scores, directions, masks, CSVs) |
| `/mnt/projects/breast_cancer/bc/dataframe.csv` | clinical table: `case_id, slide_id, label, slide_path, side, morphology, ...` for ALL train slides; morphology format `"8 500"` / `"8 520"` |

Slide names look like `2020_08728-01-T` (record-block-stain). Morphology
matching joins on `slide_name.split('-')[0]` = record.

## Pipeline overview

Two parallel tracks share the same inputs:

### Track A — unsupervised clustering (exploratory, mostly superseded)

1. **`embeddings_to_heatmap.py`** — downloads a trained linear head from
   MLflow, scores tile embeddings → per-slide parquet `slide_name, x, y, value`
   (carcinoma probability per tile). Resumable per slide. `--min-value`
   thresholds tiles when consuming later.
2. **`heatmap_to_csv.py`** — alternative entry: rasterize MLflow carcinoma
   heatmaps to a per-tile CSV (`slide_id, slide_name, x, y, value`).
   `--min-value` keeps only carcinoma tiles.
3. **`cluster_tiles.py`** — KMeans directly on tile embeddings, k-sweep +
   silhouette, cross-tab vs morphology.
4. **`cluster_regions.py`** — carcinoma tiles (connected blobs per slide via
   `scipy.ndimage.label`) → mean embedding per region → KMeans on regions →
   cross-tab vs 8500/8520. Caches `X_regions.npy` for reuse.
5. **`slide_level_cluster.py`** — region embeddings averaged per slide,
   k-means, vs morphology.
6. **`scripts/k_sweep.py`** — refit KMeans on cached `X_norm.npy` for many k,
   carcinoma-separability per k.
7. **`scripts/eval_morphology.py`** — ARI (cluster vs morphology), per-cluster
   morphology share, chi-square test, given `assignments.parquet`.

### Track B — supervised LDA projection (current main line)

**`lda_projection.py`** — the central script, two modes:

- **Training mode** (no `--load-direction`): loads tile embeddings for slides
  in the tiles CSV, filters carcinoma tiles with `--min-tile-value`
  (**0.95 for the v4 direction**), trains LDA on 8500 vs 8520 slides,
  `--save-direction` writes a **calibrated `.npz`**
  (`direction`, `scale`, `offset`, `classes`) — scale/offset fix the global
  LDA axis so colors are comparable across slides.
  Legacy single `.npy` directions still load with a warning.
- **Scoring mode** (`--load-direction foo.npz`): scores every tile of every
  slide in `--pt-dir`, writes `lda_scores.csv`
  (`slide_name, x, y, value`). With `--morphology` it finishes with a
  generalization check: per-morphology mean score, slide-level AUC,
  accuracy at score > 0. Prints progress every 25 slides (plain prints,
  tqdm does not render in job logs).

Outputs so far: `lda_direction_v4_095_val.npz` (trained on Val, min-tile 0.95),
score CSVs for Val and Train. **v5**: direction trained on Train with
`--max-tiles-per-slide 500` (memory cap; 25M tiles x 2560d won't fit),
then Val scored as the held-out test (`scripts/run_lda_val.py`).

### Masks & reporting (both tracks)

- **`make_mask_tiff.py`** — tile score CSV → full-resolution BigTIFF mask
  (grayscale for LDA scores, WSI level-0 pixel size; opens the `.mrxs` to get
  dimensions). One slide per invocation; `scripts/run_masks.py` loops a whole
  CSV on the cluster. Toggling two-category display (hide middle) is done in
  the report config, not here.
- **`make_masks.py`** — the older all-in-one: builds slide→parquet index,
  paints cluster masks (uint8 label = cluster id + 1), writes xOpat report
  config. Resumable per slide.
- **`write_masks.py`** — cluster masks from a finished `cluster_regions.py`
  run (regions.parquet).
- **`make_report_config.py` / `make_report_config_lda.py`** — standalone xOpat
  report YAML from a masks dir (WSI paths derived from the bioptic tree
  layout). `_lda` variant is for continuous grayscale LDA masks.
- **`show_masks.py`** — mask TIFFs → small JPEG previews.
- **`make_montage.py`** — per-cluster 4×4 "typical tiles" montages for the
  pathologist read.
- **`make_slide_list.py`**, **`match_morphology.py`**, **`check_csv.py`** —
  small join/validate helpers for slide lists and morphology CSVs.

## Cluster jobs

`scripts/run_*.py` are thin `kube_jobs.submit_job(...)` wrappers, all
following the same pattern: clone repo on the pod, `uv sync`, run one pipeline
script, storage mounts
`storage.secure.DATA + PROJECTS + BIOPTIC_TREE` (the storage imports at the
top are those mounts — the job needs all three).

| Script | Runs |
|---|---|
| `run_heatmaps.py` / `run_heatmaps_test.py` | `embeddings_to_heatmap.py` (Train / small test) |
| `run_clustering.py` (+ `run_clustering.sh`) | `cluster_tiles.py` k-sweep on Val |
| `run_lda.py` | `lda_projection.py` training on Train (v5) |
| `run_lda_val.py` | `lda_projection.py` scoring Val with the v5 direction |
| `run_lda_score.py` | `lda_projection.py` scoring Train with the saved direction |
| `run_masks.py` | `make_mask_tiff.py` loop over a scores CSV |
| `eval_morphology.py`, `k_sweep.py` | local utilities (not jobs) |

Monitoring: pods land in namespace `rationai-jobs-ns`. The local kubeconfig's
user can only list pods in that namespace (no nodes / cluster scope).
`scripts/pods_csv.sh [out.csv]` dumps Running/Pending pods with GPU
count/type as CSV. H100 pods are identified by nodeSelector
`nvidia.com/gpu.product=NVIDIA-H100-NVL` (nodes can't be read directly).

## Working rules (learned the hard way)

- Job pods clone from GitHub — **commit + push before submitting**.
- tqdm doesn't render in job logs — use plain `print` progress in anything
  cluster-run.
- All heavy scripts are **resumable** (skip existing per-slide outputs).
  Keep it that way when editing.
- LDA direction versioning matters: file name encodes version + min-tile
  threshold (`v4_095`). Scoring must use the same `--min-tile-value` as
  training.
- Morphology column can be `"8 500"` (space) or `"8500"` — parsers accept
  both; slide matching uses the record prefix before the first `-`.
