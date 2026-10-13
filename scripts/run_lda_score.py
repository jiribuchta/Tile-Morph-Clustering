from kube_jobs import storage, submit_job

PT_DIR = "/mnt/projects/breast_cancer/bc/tile_embeddings/virchow2/MMCI B20-24 Train"
TILES = "/mnt/projects/breast_cancer/bc/heatmaps/virchow2_heatmaps_train"
DIRECTION = "/mnt/projects/breast_cancer/tile_morph_clustering/lda_direction_v3.npz"
# CSV with slide_name + morphology columns for ALL train slides (for evaluation)
MORPH = "/mnt/projects/breast_cancer/tile_morph_clustering/train_morphology.csv"
OUT = "/mnt/projects/breast_cancer/tile_morph_clustering/train_scores.csv"

submit_job(
    job_name="tile-morph-lda-train",
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
        f"--load-direction {DIRECTION} --min-tile-value 0.9 "
        f"--morphology {MORPH} --out {OUT}",
    ],
    storage=[storage.secure.DATA, storage.secure.PROJECTS, storage.secure.BIOPTIC_TREE],
)
