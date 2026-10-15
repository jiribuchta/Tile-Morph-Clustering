from kube_jobs import storage, submit_job


# Score Val with the v5 direction (trained on Train) -> held-out generalization check
PT_DIR = "/mnt/projects/breast_cancer/bc/tile_embeddings/virchow2/MMCI B20-24 Val"
TILES = "/mnt/projects/breast_cancer/bc/heatmaps/virchow2_heatmaps_val"
DIRECTION = "/mnt/projects/breast_cancer/tile_morph_clustering/lda_direction_v5_095_train.npz"
MORPH = "/mnt/projects/breast_cancer/bc/dataframe.csv"
OUT = "/mnt/projects/breast_cancer/tile_morph_clustering/lda_scores_v5_095_val.csv"

submit_job(
    job_name="tile-morph-lda-v5-val",
    username="jiribuchta",
    image="cerit.io/rationai/base:2.0.6",
    cpu=8,
    memory="16Gi",
    public=False,
    script=[
        "git clone https://github.com/jiribuchta/Tile-Morph-Clustering.git workdir",
        "cd workdir",
        "uv sync",
        f"uv run python -u lda_projection.py --pt-dir '{PT_DIR}' --tiles {TILES} "
        f"--load-direction {DIRECTION} --min-tile-value 0.95 "
        f"--morphology {MORPH} --out {OUT}",
    ],
    storage=[storage.secure.DATA, storage.secure.PROJECTS, storage.secure.BIOPTIC_TREE],
)
