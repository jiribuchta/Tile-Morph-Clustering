from kube_jobs import storage, submit_job


submit_job(
    job_name="tile-morph-clustering-masks",
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
        "uv run -m make_masks +data=mmci_b20_24_train +experiment/masks=mammaprint mode=full"
    ],
    storage=[storage.secure.DATA, storage.secure.PROJECTS, storage.secure.BIOPTIC_TREE],
)
