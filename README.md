# Tile-Morph-Clustering

Tile-level morphology analysis of breast-cancer whole-slide images (MAMMAPRINT
project). The one question everything here answers: **can tile embeddings
(Virchow2) separate ductal (ICD-O 8500) from lobular (8520) carcinoma
morphology?**

Short answer: yes. A linear projection trained on carcinoma tiles ranks
held-out slides with **AUC 0.972, 94.4% accuracy** (see
[PATHOLOGIST_REPORT.md](PATHOLOGIST_REPORT.md), Czech version in
`PATHOLOGIST_REPORT_CZ.md`).

## How the pipeline works

Everything starts from per-slide `.pt` files with 2560-d L2-normalized
Virchow2 tile embeddings (one row per 224-px tile; tile coords are in
level-1 pixel space). The current main line:

1. **Carcinoma heatmaps** — `embeddings_to_heatmap.py` downloads a trained
   linear head from MLflow and scores every tile's carcinoma probability,
   writing a per-slide parquet (`slide_name, x, y, value`).
2. **LDA direction** — `lda_projection.py` (training mode) loads embeddings
   for slides in a tiles CSV, keeps only carcinoma tiles
   (`--min-tile-value 0.95`), trains LDA on 8500-vs-8520 slides and saves a
   calibrated `.npz` (`direction, scale, offset, classes`). The calibration
   fixes the axis globally: score 0 = decision boundary, same meaning on
   every slide. Negative = ductal-like, positive = lobular-like.
3. **Scoring** — `lda_projection.py` (scoring mode, `--load-direction`)
   scores every tile of every slide in the embeddings dir and writes
   `lda_scores.csv`, ending with a slide-level generalization check
   (`evaluate_scores`): per-morphology means, AUC, accuracy — excluding
   slides with fewer than `--min-tiles` (default 100) scored tiles, since
   their mean is noise (mostly negative blocks).
4. **Offline eval** — `eval_lda_scores.py` re-runs the same check on an
   existing scores CSV without touching the embeddings.
5. **Masks** — `make_mask_tiff.py` rasterizes a scores CSV into a
   full-resolution grayscale BigTIFF (blue/red via the report's colormap;
   score ±4 maps to 0–255). One slide per call; `scripts/run_masks.py`
   loops a whole CSV on the cluster.
6. **xOpat report** — `make_report_config_lda.py` builds the report YAML
   from a masks dir (the generated configs live on the cluster under
   `tile_morph_clustering/`); submit the serving job from there.

`make_heatmap_thumb.py` and `show_masks.py` (quick previews) and the
slide-list/morphology helpers were removed with the k-means era — see git
history if you need them back.

## Data locations (cluster mounts)

| Path | Contents |
|---|---|
| `/mnt/projects/breast_cancer/bc/tile_embeddings/virchow2/MMCI B20-24 Train` / `... Val` | per-slide Virchow2 `.pt` embeddings |
| `/mnt/projects/breast_cancer/bc/heatmaps/...` | carcinoma-probability parquets |
| `/mnt/projects/breast_cancer/bc/dataframe.csv` | clinical table (label, morphology — `8500`/`8 520` formats) |
| `/mnt/bioptic_tree/YYYY/MM/DD/<slide>/<slide>.mrxs` | whole-slide images; slide name = `record-block-suffix`, suffix `T`=tumor, `N`=negative, `M`=metastases |
| `/mnt/projects/breast_cancer/tile_morph_clustering/` | all pipeline outputs (directions, score CSVs, masks, reports) |

Morphology matching joins on `slide_name.split('-')[0]` (the record) — beware
that `N` blocks inherit the case label this way.

## Running on the cluster

Jobs run on the CERIT Kubernetes cluster. The `scripts/run_*.py` wrappers all
follow the same pattern: clone this repo on the pod, `uv sync`, run one
pipeline script, with the DATA/PROJECTS/BIOPTIC_TREE mounts attached.
**Commit + push before submitting** — pods clone from GitHub.

Everything cluster-run uses plain `print` progress (tqdm doesn't render in
job logs) and is **resumable** per slide — keep both properties when editing.

Monitoring: pods land in namespace `rationai-jobs-ns`.

## Layout notes

- `ml/`, `configs/`, `preprocessing/` — PyTorch Lightning + Hydra + MLflow
  template scaffolding, currently unused by the pipeline but kept for
  planned integration.
- The unsupervised k-means track (tile/region/slide clustering and its
  mask/report tools) was removed — git history has it if ever needed.

## Results so far

- `lda_direction_v5_095_train.npz` — trained on Train, min-tile 0.95,
  500-tile cap per slide (memory).
- Val scoring (`lda_scores_v5_095_val.csv`): 196 well-sampled labeled slides
  → AUC 0.972, acc@0 94.4%, balanced 95.3%. 97 excluded slides are mostly
  negative (`N`) blocks — detector correctly finds no tumor there.
- Two genuine contradictions for pathologist review:
  `2020_08267-04-T2` (8520, scores −1.48) and `2023_08660-2-T`
  (8500, scores +1.63).
