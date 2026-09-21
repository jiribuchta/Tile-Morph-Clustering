from kube_jobs import storage, submit_job


submit_job(
    job_name="tile-morph-clustering",
    username="jiribuchta",
    image="cerit.io/rationai/base:2.0.6",
    cpu=8,
    memory="32Gi",  # X ~ 1842 slides x 256 tiles x 2560-dim float32 ~= 5GiB
    public=False,
    script=[
        "git clone https://github.com/jiribuchta/Tile-Morph-Clustering.git workdir",
        "cd workdir",
        "uv sync",
        "export MLFLOW_TRACKING_URI=http://mlflow.rationai-mlflow:5000/",
        "uv run python -c 'from mlflow.tracking import MlflowClient; e = MlflowClient().get_experiment_by_name(\"Breast Cancer\"); assert e, \"no Breast Cancer exp\"; print(\"tracking OK, exp\", e.experiment_id)'",
        "uv run -m cluster_tiles +data=mmci_b20_24 +experiment/clustering=train_k32",
    ],
    storage=[storage.secure.DATA, storage.secure.PROJECTS],
)
