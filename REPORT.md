# Tile-Morph-Clustering — pipeline report

Goal: find out whether carcinoma morphologies (8500 / 8504 / 8520) form
separate clusters in the Virchow2 tile embeddings, using the carcinoma
heatmaps as the source of truth for *where* the cancer is.

## Data flow

```
WSI (.mrxs)
  │
  ├─► tile embeddings (Virchow2, 2560-d)          [existing: tiling_parquets_sharded]
  │       MMCI B20-24 {Train,Val}/Virchow2/tiles/*.parquet
  │       schema: slide_id (32-byte hex), x, y (level-1 px), carcinoma, embedding
  │
  └─► carcinoma heatmaps (one .tiff per slide)     [existing: mlflow run, exp 61]
              │
              ▼
      1. heatmap_to_csv.py      heatmaps + slides.parquet ──► heatmaps_tiles.csv
              │                     (slide_id, slide_name, x, y, value; value >= 0.5)
              ▼
      2. match_morphology.py    slides.txt + morphology.csv ──► slide_morphology.csv
              │                     (slide_name, record, morphology)
              ▼
      3. cluster_regions.py     heatmaps_tiles.csv + slide_morphology.csv
              │                  + tiling parquets ──► region clustering
              ▼
      4. make_masks.py          clustering output + WSIs ──► WSI-level BigTIFF masks
              │                  + xOpat report yaml
              ▼
      5. make_montage.py        clustering output + WSIs ──► per-cluster tile montages
                                     (for the pathologist read)
```

## The scripts

### 1. `heatmap_to_csv.py` — heatmaps → per-tile values

Downloads the carcinoma heatmaps + Val `slides.parquet` from mlflow, computes the
mean heatmap value per tile (224×224 tiles, stride 112, level-1 grid), writes one
CSV row per carcinoma tile.

- `--min-value 0.5` → only carcinoma tiles (the 650 MB file, 6.85 M rows, 292 slides)
- Speed trick: 2×2 block-average before the integral-image pass (¼ the pixels,
  ~6× faster); tile means preserved up to a 0.5/255 floor error
- Streams per slide to disk — never holds a full grid in RAM
- **Every slide in the CSV is complete** (one `to_csv` per slide), so a crash only
  loses *missing* slides, never partial ones

### 2. `match_morphology.py` — slide names → morphology labels

Joins a plain list of slide names to the pathology morphology CSV on
`record = slide_name.split("-")[0]` (e.g. `2020_00172-04-T` → `2020_00172`).

- flags malformed lines (e.g. two names concatenated)
- shows left/right morphology conflicts as `left:8520 | right:8500`
- lists slides with no morphology record

### 3. `cluster_regions.py` — the core: region-level clustering

The question "do 8500 and 8520 cluster separately?" answered at **region** level
(one connected carcinoma blob = one sample), not tile level.

Pipeline inside:
1. **regions** — stream the tiles CSV, keep `value >= --min-value`, group per slide
   into 8-connected blobs (`scipy.ndimage.label`), drop blobs < `--min-region` tiles
2. **index** — map `slide_id` → parquet part, reading *only* the `slide_id` column
   (cheap); only the parts containing our slides are streamed
3. **collect** — one pass over those parts: each carcinoma tile's L2-normalized
   embedding is added to its region's running sum (no per-tile storage)
4. **cluster** — one vector per region (mean of its tiles), KMeans on regions
5. **report** — morphology vs majority-cluster cross-tab (per slide)

Outputs in `--out`:
| file | what |
|---|---|
| `regions.parquet` | per region: slide_id, region, n_tiles, cx, cy, cluster, morphology |
| `region_meta.parquet` | same minus cluster (k-independent, used by `--load-x`) |
| `X_regions.npy` | region feature vectors (k-independent, cached) |
| `centroids.npy` | KMeans centroids |
| `summary.json` | morphology × cluster cross-tab + metrics |
| `masks/<slide>.tiff` | tile-grid mask: 0 = background, cluster c → c+1 |

`--load-x` reloads `X_regions.npy` + `region_meta.parquet` and skips the ~30 min
embedding collection — changing `--k` then takes seconds.

**Result (k=2, 292 slides, 13 440 regions):** morphologies do *not* separate
cleanly — 8500 splits 64/175 across the two clusters, silhouette 0.109. The
dominant variance is per-slide, not morphology.

### 4. `make_masks.py` — WSI-level masks for xOpat

Takes the clustering output and writes one BigTIFF mask per slide at **WSI
level-0 size** (pixel value = cluster id + 1, 0 = background), so each mask
overlays its WSI 1:1 in xOpat. Also writes the xOpat report config yaml.

- each tile → nearest centroid (same KMeans run → `centroids.npy`)
- resumable (existing .tiff skipped), failure-isolated (bad WSI logged, not fatal)
- slide_id → parts index is cached at `out/slide_parts_index.json`
- hydra: `uv run -m make_masks +data=mmci_b20_24_train +experiment/masks=mammaprint`

### 5. `make_montage.py` — the pathologist read

For each cluster, picks the tiles **closest to that cluster's centroid** (the
typical members) and crops them from the WSIs into one 4×4 PNG per cluster.
The pathologist reads the images to decide which clusters are coherent
morphologies and which are mixes. No re-clustering — just similarity to
`centroids.npy`.

### Utilities

- `check_csv.py` — which slides are in a (possibly partial) tiles CSV, with row counts
- `split_slides.py` — filter a slides CSV to records with prefix 850/852
- `visualize_mask.py` — ad-hoc: open a mask tiff with openslide, save a thumbnail
- `cluster_tiles.py` — the *original* tile-level clustering (k=32 morphology
  discovery), hydra-based; see `README2.md`

## Key formats & gotchas

- **slide_id**: 32 raw bytes in the parquets → 64-char hex string in CSVs.
  Both sides use `.hex()`; a format mismatch silently yields 0 matched slides.
- **Val vs Train**: the heatmap CSV comes from the **Val** slides
  (`heatmap_to_csv.py` downloads the Val slides.parquet) → point `--embeddings`
  at `MMCI B20-24 Val/Virchow2`, not Train.
- **coords**: tile `x, y` are level-1 pixel top-left, tile 224×224, stride 112.
  `x * 1_000_000 + y` is the unique tile key used for the join.
- **region label 0** = tile in a dropped (too-small) region — skipped in collection.
- **numpy `&` does not short-circuit** — the searchsorted guard must be two-step
  (see `cluster_regions.py` collect loop).

## Running the pipeline (cluster)

```bash
# 1. tiles CSV (once; ~3 h for 292 slides)
uv run python -u heatmap_to_csv.py --out heatmaps_tiles.csv --min-value 0.5

# 2. morphology labels (seconds)
uv run python -u match_morphology.py slides.txt morphology.csv --out slide_morphology.csv

# 3. region clustering (~30 min; then --load-x for other k)
uv run python -u cluster_regions.py \
    --tiles heatmaps_tiles.csv \
    --morphology slide_morphology.csv \
    --embeddings "/mnt/projects/breast_cancer/bc/tiling_parquets_sharded/MMCI B20-24 Val/Virchow2" \
    --min-value 0.5 --min-region 8 --k 2 \
    --out /mnt/projects/breast_cancer/tile_morph_clustering/region_clustering_1

# 3b. try other k (seconds, reuses cached X)
uv run python -u cluster_regions.py --load-x \
    --tiles heatmaps_tiles.csv --morphology slide_morphology.csv \
    --min-value 0.5 --min-region 8 --k 3 \
    --out /mnt/projects/breast_cancer/tile_morph_clustering/region_clustering_1

# 4. WSI-level masks + xOpat report
uv run -m make_masks +data=mmci_b20_24_train +experiment/masks=mammaprint

# 5. montages for the pathologist
uv run -m make_montage ...
```
