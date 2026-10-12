from kube_jobs import storage, submit_job

PT_DIR = "/mnt/projects/breast_cancer/bc/tile_embeddings/virchow2/MMCI B20-24 Train"
OUT = "/mnt/projects/breast_cancer/bc/heatmaps/test_one_slide"

submit_job(
    job_name="tile-morph-heatmaps-test",
    username="jiribuchta",
    image="cerit.io/rationai/base:2.0.6",
    cpu=2,
    memory="8Gi",
    public=False,
    script=[
        "git clone https://github.com/jiribuchta/Tile-Morph-Clustering.git workdir",
        "cd workdir",
        "uv sync",
        "echo 'slide_name\\n2019_08728-01-T' > one_slide.csv",
        f"uv run python -u embeddings_to_heatmap.py --pt-dir '{PT_DIR}' --slides one_slide.csv --out {OUT} --min-value 0.5",
        f"ls -la {OUT}",
    ],
    storage=[storage.secure.DATA, storage.secure.PROJECTS, storage.secure.BIOPTIC_TREE],
)
