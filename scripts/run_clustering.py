from kube_jobs import storage, submit_job


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
        "uv run cluster_regions.py --out /mnt/projects/breast_cancer/tile_morph_clustering/region_clustering_1 --tiles /mnt/projects/breast_cancer/tile_morph_clustering/heatmaps_tiles.csv --morphology /mnt/projects/breast_cancer/tile_morph_clustering/slide_morphology.csv --embeddings \"/mnt/projects/breast_cancer/bc/tiling_parquets_sharded/MMCI B20-24 Train/Virchow2\" --min-value 0.5 --min-region 8 --k 2",
    ],
    storage=[storage.secure.DATA, storage.secure.PROJECTS],
)
