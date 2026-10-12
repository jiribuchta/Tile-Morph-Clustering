"""Generate xOpat report config for LDA morphology masks (continuous grayscale).

Usage:
    python make_report_config_lda.py --masks-dir /path/to/masks \
        --wsi-root /mnt/bioptic_tree \
        [--out /path/to/report_conf/reporter/lda_morphology.yaml]
"""

import argparse
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--masks-dir", required=True, help="dir containing .tiff mask files")
    ap.add_argument("--wsi-root", default="/mnt/bioptic_tree")
    ap.add_argument("--out", default=None, help="output YAML path")
    args = ap.parse_args()

    masks_dir = Path(args.masks_dir).resolve()
    tiffs = sorted(masks_dir.glob("*.tiff"))
    if not tiffs:
        raise SystemExit(f"no .tiff files in {masks_dir}")

    names = [t.stem.replace("mask_", "") for t in tiffs]

    cfg = f"""# xOpat report config: LDA morphology masks (continuous grayscale)
#
# Mask files are uint8 grayscale (0-255) where:
#   low values = 8500-leaning, high values = 8520-leaning
#
# Run:
#     python -m report --config-dir report_conf reporter=lda_morphology user=<you>

defaults:
  - _self_

title: LDA Morphology Projection - MAMMAPRINT

background:
  _target_: report.masks.BasicImageRetriever
  source_dir: /mnt/data/MOU/breast/mammaprint
  globs: ["*.mrxs"]
  layer_name: MAMMAPRINT WSI (MIRAX)

mask_retrievers:
  - _target_: report.masks.DriveMasksRetriever
    dir_name: {masks_dir}
    layer_name: LDA Morphology Score
    shader_conf:
      type: colormap
      params:
        use_channel0: r
        opacity: 0.55
        connect:
          type: bool
          interactive: false
          default: true
        color:
          type: custom_colormap
          default:
          - "#2166AC"   # 0 = 8500 (blue)
          - "#43A2CA"   # 64
          - "#F7F7F7"   # 128 (neutral)
          - "#FCBBA1"   # 192
          - "#B2182B"   # 255 = 8520 (red)
          steps: 5
          mode: qualitative
          interactive: true
          title: 'Morphology:'
          continuous: true
        threshold:
          breaks:
          - 0.0
          - 1.0
          mask:
          - 1
          - 1
          interactive: false
          inverted: false
          maskOnly: false
          toggleMask: false
          title: Range
          min: 0
          max: 1
          minGap: 0.1
          step: null
          pips:
            mode: range
            values: 2
            density: 4
            stepped: false

save:
  _target_: report.save.MLFlowReportStorer
  artifact_path: report.html
  metadata:
    experiment_name: Breast Cancer
    run_name: LDA morphology masks
    description: LDA morphology projection masks overlaid on MAMMAPRINT WSIs
    user: ${{user}}
"""

    out = Path(args.out) if args.out else masks_dir.parent / "report_conf" / "reporter" / "lda_morphology.yaml"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(cfg)
    print(f"wrote {out} ({len(names)} slides)")


if __name__ == "__main__":
    main()
