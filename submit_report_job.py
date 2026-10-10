"""Submit the xOpat report job.

The report config is written by make_masks.py to
/mnt/projects/breast_cancer/tile_morph_clustering/masks_k32_v2/report_conf/reporter/tile_morph_k32.yaml
This job clones the report tool, installs it, and runs it with that config.
The HTML report is saved to the MLflow "Breast Cancer" experiment
(see save: in tile_morph_k32.yaml).
"""
from kube_jobs import storage, submit_job

submit_job(
    job_name="xopat-report-tile-morph",
    username="jiribuchta",
    image="cerit.io/rationai/base:2.0.6",
    cpu=4,
    memory="16Gi",
    public=False,
    script=[
        "git clone git@gitlab.ics.muni.cz:rationai/digital-pathology/tools/report.git workdir",
        "cd workdir",
        "pip install -e .",
        "python -m report --config-dir /mnt/projects/breast_cancer/tile_morph_clustering/region_clustering_1/report_conf/reporter +reporter=region_clustering user=jiribuchta",
    ],
    storage=[storage.secure.DATA, storage.secure.PROJECTS],
)
