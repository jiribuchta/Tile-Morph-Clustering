"""Pinpoint why medoid crops are black: alpha? coords? level?

For each medoid slide (up to --max), opens the .mrxs once and:
  - reports alpha / channel stats of the nominal crop (RGBA)
  - renders a labeled overview: tissue bbox + medoid crop boxes drawn on it
  - prints where tissue actually is (non-black bbox at the crop's level)

Writes ``overview_<slide>.png`` per slide. Run on the WSI node:
    uv run python scripts/thumbnail_medoids.py \
        --medoids <out>/medoids.jsonl --out ./medoids --max 4
"""

import argparse
import json
from pathlib import Path

import numpy as np
import openslide
from PIL import Image, ImageDraw


def _bbox_nonblack(arr: np.ndarray, thr: int = 16):
    """Bounding box of non-black pixels in an HxWx* grayscale-ish array."""
    g = arr.max(axis=-1) if arr.ndim == 3 else arr
    ys, xs = np.where(g > thr)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def medoid_overview(r: dict, out: Path, i: int) -> None:
    p = r["slide_path"]
    lv, x, y, w, h = r["level"], r["x"], r["y"], r["w"], r["h"]
    with openslide.OpenSlide(p) as s:
        # nominal crop RGBA
        crop = s.read_region((x, y), lv, (w, h))
        carr = np.asarray(crop)
        alpha = (
            f"alpha min={carr[..., 3].min()} max={carr[..., 3].max()}"
            if carr.shape[-1] == 4 else "no alpha"
        )
        rgb_mean = float(np.asarray(crop.convert("RGB")).mean())

        # overview at ~32x downsample, with markers
        best = s.get_best_level_for_downsample(32.0)
        W, H = s.level_dimensions[best]
        ov = s.read_region((0, 0), best, (W, H)).convert("RGB")
        ov2l0 = s.level_dimensions[0][0] / ov.width  # overview px -> level-0 px

        # tissue bbox at this overview level
        tbb = _bbox_nonblack(np.asarray(ov))

        draw = ImageDraw.Draw(ov)
        notes = [f"c={r['cluster']} L{lv} ({x},{y}) cropRGBmean={rgb_mean:.0f}"]
        if tbb:
            x0, y0, x1, y1 = tbb
            draw.rectangle([x0, y0, x1, y1], outline="lime", width=2)
            notes.append(f"tissue_bbox_L0~({int(x0*ov2l0)},{int(y0*ov2l0)})-({int(x1*ov2l0)},{int(y1*ov2l0)})")
        # medoid box (convert level-lv coords to overview coords)
        lv_w, lv_h = s.level_dimensions[lv]
        kx = ov.width / lv_w
        ky = ov.height / lv_h
        bx0, by0 = int(x * kx), int(y * ky)
        bx1, by1 = int((x + w) * kx), int((y + h) * ky)
        draw.rectangle([bx0, by0, bx1, by1], outline="red", width=2)
        draw.line([bx0, by0, bx1, by1], fill="red", width=1)

        if ov.width > 1024:
            rr = 1024 / ov.width
            ov = ov.resize((1024, int(ov.height * rr)))

    print(f"[{i}] {Path(p).name} L{lv} ({x},{y}) {w}x{h}")
    print(f"    crop: {alpha}, RGBmean={rgb_mean:.0f}")
    print(f"    {' | '.join(notes)}")
    ov.save(out / f"overview_{Path(p).stem}_{i}.png")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--medoids", required=True)
    ap.add_argument("--out", default="./medoids")
    ap.add_argument("--max", type=int, default=4)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    with open(args.medoids) as f:
        rows = [json.loads(l) for l in f if l.strip()][: args.max]
    for i, r in enumerate(rows):
        p = Path(r["slide_path"])
        if not p.exists():
            print(f"[{i}] MISSING {p}")
            continue
        medoid_overview(r, out, i)
    print(f"\nwrote overviews to {out}")


if __name__ == "__main__":
    main()
