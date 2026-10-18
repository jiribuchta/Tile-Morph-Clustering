from kube_jobs import storage, submit_job


TILES = "/mnt/projects/breast_cancer/tile_morph_clustering/lda_scores_v6_095_test.csv"
WSI_ROOT = "/mnt/bioptic_tree"
OUT_DIR = "/mnt/projects/breast_cancer/tile_morph_clustering/masks_v6_095_test"
MORPH = "/mnt/projects/breast_cancer/bc/dataframe.csv"  # warn-only: unlabeled slides still get masks

submit_job(
    job_name="tile-morph-masks-test",
    username="jiribuchta",
    image="cerit.io/rationai/base:2.0.6",
    cpu=8,
    memory="16Gi",
    public=False,
    script=[
        "git clone https://github.com/jiribuchta/Tile-Morph-Clustering.git workdir",
        "cd workdir",
        "uv sync",
        f"uv run python -u make_mask_tiff.py --tiles {TILES} --wsi-root {WSI_ROOT} "
        f"--slide all --out {OUT_DIR} --morphology {MORPH}",
    ],
    storage=[storage.secure.DATA, storage.secure.PROJECTS, storage.secure.BIOPTIC_TREE],
)
