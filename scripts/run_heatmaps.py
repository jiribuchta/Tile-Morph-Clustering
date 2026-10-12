from kube_jobs import storage, submit_job

PT_DIR = "/mnt/projects/breast_cancer/bc/tile_embeddings/virchow2/MMCI B20-24 Val"
OUT = "/mnt/projects/breast_cancer/tile_morph_clustering/heatmaps_val"

submit_job(
    job_name="tile-morph-heatmaps",
    username="jiribuchta",
    image="cerit.io/rationai/base:2.0.6",
    cpu=8,
    memory="32Gi",
    public=False,
    script=[
        "git clone https://github.com/jiribuchta/Tile-Morph-Clustering.git workdir",
        "cd workdir",
        "uv sync",
        f"uv run python -u embeddings_to_heatmap.py --pt-dir '{PT_DIR}' --out {OUT} --min-value 0.5",
    ],
    storage=[storage.secure.DATA, storage.secure.PROJECTS, storage.secure.BIOPTIC_TREE],
)
