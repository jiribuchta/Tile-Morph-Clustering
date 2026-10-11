from kube_jobs import storage, submit_job

OUT = "/mnt/projects/breast_cancer/tile_morph_clustering/tile_clustering"
TILES = "/mnt/projects/breast_cancer/tile_morph_clustering/test_heatmaps.csv"
MORPH = "/mnt/projects/breast_cancer/tile_morph_clustering/slide_list_balanced.csv"
PT_DIR = "/mnt/projects/breast_cancer/bc/tile_embeddings/virchow2/MMCI B20-24 Val"

submit_job(
    job_name="tile-morph-clustering",
    username="jiribuchta",
    image="cerit.io/rationai/base:2.0.6",
    cpu=8,
    memory="64Gi",
    public=False,
    script=[
        "git clone https://github.com/jiribuchta/Tile-Morph-Clustering.git workdir",
        "cd workdir",
        "uv sync",
        f"uv run python -u cluster_tiles.py --pt-dir '{PT_DIR}' --tiles {TILES} --morphology {MORPH} --ks 2,4,8,16,32 --max-tiles 2000000 --out {OUT}",
    ],
    storage=[storage.secure.DATA, storage.secure.PROJECTS, storage.secure.BIOPTIC_TREE],
)
