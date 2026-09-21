from kube_jobs import storage, submit_job


submit_job(
    job_name="project_name-clustering",
    username=...,
    image="cerit.io/rationai/base:2.0.6",
    cpu=8,
    memory="32Gi",  # X ~ 1842 slides x 256 tiles x 2560-dim float32 ~= 5GiB
    public=False,
    script=[
        "git clone https://gitlab.ics.muni.cz/rationai/digital-pathology/pathology/project_name workdir",
        "cd workdir",
        "uv sync",
        "export MLFLOW_USER=<user> && uv run -m cluster_tiles +data=mmci_b20_24 +experiment/clustering=train_k32",
    ],
    storage=[storage.secure.DATA],
)
