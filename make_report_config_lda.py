"""Generate xOpat report config for LDA morphology masks (continuous grayscale).

Usage:
    python make_report_config_lda.py --masks-dir /path/to/masks \
        --wsi-root /mnt/bioptic_tree \
        [--out /path/to/report_conf/reporter/lda_morphology.yaml]
"""

import argparse
from pathlib import Path


def wsi_path(name: str, root: str) -> str:
    """Derive WSI path from slide name (bioptic tree structure)."""
    year = name[:4]
    mmddd = name[5:10]
    month = mmddd[:2]
    day = mmddd[2:]
    return f"{root}/{year}/{month}/{day}/{name}/{name}.mrxs"


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
    paths = [wsi_path(n, args.wsi_root) for n in names]
    paths_block = "\n".join(f"    - {p}" for p in paths)

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
  _target_: report.masks.SlideRetriever
  paths:
{paths_block}
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
          - ''             # value 0 = background (no tiles)
          - "#2166AC"      # low = 8500 (ductal)
          - "#F7F7F7"      # middle = ambiguous
          - "#B2182B"      # high = 8520 (lobular)
          steps: 4
          mode: qualitative
          interactive: true
          title: 'Morphology:'
          continuous: false
        threshold:
          breaks:
          - 0.003922     # value 0 = background
          - 0.43          # below ~ -0.6 LDA units = 8500
          - 0.57          # above ~ +0.6 = 8520
          mask:
          - 0            # background: hidden
          - 1            # 8500: shown
          - 0            # middle: hidden by default
          - 1            # 8520: shown
          interactive: false
          inverted: true
          maskOnly: true
          toggleMask: true
          title: Categories
          min: 0
          max: 1
          minGap: 0.05
          step: null
          pips:
            mode: count
            values: 4
            density: 4
            stepped: true

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
