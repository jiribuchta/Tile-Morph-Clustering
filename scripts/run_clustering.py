from kube_jobs import storage, submit_job

OUT = "/mnt/projects/breast_cancer/tile_morph_clustering/region_clustering_1"
TILES = "/mnt/projects/breast_cancer/tile_morph_clustering/heatmaps_tiles.csv"
MORPH = "/mnt/projects/breast_cancer/tile_morph_clustering/slide_morphology.csv"
EMBED = "/mnt/projects/breast_cancer/bc/tiling_parquets_sharded/MMCI B20-24 Val/Virchow2"
K = 32

submit_job(
    job_name="tile-morph-clustering",
    username="jiribuchta",
    image="cerit.io/rationai/base:2.0.6",
    cpu=8,
    memory="32Gi",
    public=False,
    script=[
        "git clone https://github.com/jiribuchta/Tile-Morph-Clustering.git workdir",
        "cd workdir",
        "uv sync",
        "export MLFLOW_TRACKING_URI=http://mlflow.rationai-mlflow:5000/",
        "uv run python -c 'from mlflow.tracking import MlflowClient; e = MlflowClient().get_experiment_by_name(\"Breast Cancer\"); assert e, \"no Breast Cancer exp\"; print(\"tracking OK, exp\", e.experiment_id)'",
        # 1. clustering + evaluation
        f"uv run python -u cluster_regions.py --out {OUT} --tiles {TILES} --morphology {MORPH} --embeddings '{EMBED}' --min-value 0.5 --min-region 8 --k {K}",
        # 2. masks + report config
        f"uv run python -u write_masks.py --out {OUT} --tiles {TILES} --min-value 0.5 --min-region 8 --k {K} --wsi-root /mnt/bioptic_tree",
    ],
    storage=[storage.secure.DATA, storage.secure.PROJECTS, storage.secure.BIOPTIC_TREE],
)
