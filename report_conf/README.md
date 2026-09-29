# TileMorph k=32 masks + xOpat report

Two steps, both run on the cluster where the WSIs and the Virchow2 tile
parquets are mounted.

## 1) Build the masks (this repo)

One `.tiff` per slide: uint8 label image at the WSI **level-0 size**, pixel
value = cluster id, `0` = background. Saved with `ratiopath.masks.write_big_tiff`
(512px tiles, DEFLATE, pyramid) so each mask overlays its WSI 1:1 in xOpat.

Smoke test (paint only the tiles `run_clustering` already assigned — fast, no
re-embedding), first 5 slides:

    uv run -m make_masks mode=sampled slides=5 +experiment/masks=mammaprint

Full run (every slide, every tile re-assigned to the nearest centroid — needs
the dataset to stream the tile parquets):

    uv run -m make_masks mode=full slides=0 +data=mmci_b20_24_train +experiment/masks=mammaprint

Outputs go to `out` (default `${project_path}/masks_k32`), one `<slide>.tiff`
plus a `manifest.json`. After each save the mask is re-opened and every
painted tile's corner pixel is checked to equal its cluster (fails the job on
misalignment).

## 2) Build the report (the `report` tool)

`report_conf/` is a ready-to-use Hydra config dir for the `report` package
(https://gitlab.ics.muni.cz/rationai/digital-pathology/pipeline/report).

    # one-time: install the report tool in its own env
    git clone https://gitlab.ics.muni.cz/rationai/digital-pathology/pipeline/report.git
    cd report && pdm install && cd ..

    # generate the report (from this repo root)
    cd report
    pdm report --config-dir ../report_conf reporter=tile_morph_k32 user=<your_name>
    # the `mlflow` tracking group defaults to `kubas_cluster` (in-cluster);
    # override outside the cluster, e.g. mlflow=kubas_external  (or mlflow=local)

This reads the WSIs (`.mrxs`) as the background, overlays the masks by matching
filename stem, and stores an MLflow run (`Breast Cancer` experiment) with
`report.html`. Open the run in xOpat: each slide shows the WSI with the cluster
mask layered on top (custom `colormap`; cluster 0 transparent, one colour per
cluster id).

### If a path differs
Edit `report_conf/reporter/tile_morph_k32.yaml`:
- `background.source_dir` — WSI root (the `data_path` used by `run_clustering`)
- `mask_retrievers[0].dir_name` — where `make_masks` wrote the `.tiff` masks
