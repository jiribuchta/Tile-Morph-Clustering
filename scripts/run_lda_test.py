from kube_jobs import storage, submit_job


# Score the Test split with the v6 direction (held-out, same as Val check)
PT_DIR = "/mnt/projects/breast_cancer/bc/tile_embeddings/virchow2/Breast MMCI Full Test"
TILES = "/mnt/projects/breast_cancer/bc/heatmaps/virchow2_heatmaps_test"
DIRECTION = (
    "/mnt/projects/breast_cancer/tile_morph_clustering/lda_direction_v6_095_train.npz"
)
MORPH = "/mnt/projects/breast_cancer/bc/dataframe.csv"
OUT = "/mnt/projects/breast_cancer/tile_morph_clustering/lda_scores_v6_095_test.csv"

submit_job(
    job_name="tile-morph-lda-v6-test",
    username="jiribuchta",
    image="cerit.io/rationai/base:2.0.6",
    cpu="8",
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
