from kube_jobs import storage, submit_job

TILES = "/mnt/projects/breast_cancer/tile_morph_clustering/lda_scores.csv"
WSI_ROOT = "/mnt/bioptic_tree"
OUT_DIR = "/mnt/projects/breast_cancer/tile_morph_clustering/masks"

submit_job(
    job_name="tile-morph-masks",
    username="jiribuchta",
    image="cerit.io/rationai/base:2.0.6",
    cpu=8,
    memory="64Gi",
    public=False,
    script=[
        "git clone https://github.com/jiribuchta/Tile-Morph-Clustering.git workdir",
        "cd workdir",
        "uv sync",
        f"mkdir -p {OUT_DIR}",
        f"cut -d, -f1 {TILES} | tail -n +2 | sort -u | while read s; do "
        f"uv run python -u make_mask_tiff.py --tiles {TILES} --wsi-root {WSI_ROOT} --slide \"$s\" --out {OUT_DIR}/$s.tiff; "
        f"done",
    ],
    storage=[storage.secure.DATA, storage.secure.PROJECTS, storage.secure.BIOPTIC_TREE],
)
