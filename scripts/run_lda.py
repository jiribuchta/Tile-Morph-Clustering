from kube_jobs import storage, submit_job


# v6: retrain with the coord-join fix -- training now selects the actual
# carcinoma tiles (v5 paired the first N tiles of the whole slide with
# carcinoma coords, so the v5 direction and its AUC were computed misaligned)
PT_DIR = "/mnt/projects/breast_cancer/bc/tile_embeddings/virchow2/MMCI B20-24 Train"
TILES = "/mnt/projects/breast_cancer/bc/heatmaps/virchow2_heatmaps_train"
MORPH = "/mnt/projects/breast_cancer/bc/dataframe.csv"
OUT = "/mnt/projects/breast_cancer/tile_morph_clustering/lda_scores_v6_095_train.csv"
DIRECTION_OUT = "/mnt/projects/breast_cancer/tile_morph_clustering/lda_direction_v6_095_train.npz"

submit_job(
    job_name="tile-morph-lda-v6-train",
    username="jiribuchta",
    image="cerit.io/rationai/base:2.0.6",
    cpu=8,
    memory="64Gi",
    public=False,
    script=[
        "git clone https://github.com/jiribuchta/Tile-Morph-Clustering.git workdir",
        "cd workdir",
        "uv sync",
        f"uv run python -u lda_projection.py --pt-dir '{PT_DIR}' --tiles {TILES} "
        f"--morphology {MORPH} --min-tile-value 0.95 --max-tiles-per-slide 500 "
        f"--save-direction {DIRECTION_OUT} --out {OUT}",
    ],
    storage=[storage.secure.DATA, storage.secure.PROJECTS, storage.secure.BIOPTIC_TREE],
)
