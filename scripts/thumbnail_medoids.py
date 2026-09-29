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


def _stats(s, level, x, y, w, h):
    W, H = s.level_dimensions[level]
    if x >= W or y >= H:
        return f"L{level} OOB"
    w, h = min(w, W - x), min(h, H - y)
    a = np.asarray(s.read_region((x, y), level, (w, h)))
    al = f"a{a[..., 3].min()}-{a[..., 3].max()}" if a.shape[-1] == 4 else "no-a"
    return f"L{level} rgb{float(a[..., :3].mean()):.0f} {al}"


def medoid_overview(r: dict, out: Path, i: int) -> None:
    p = r["slide_path"]
    lv, x, y, w, h = r["level"], r["x"], r["y"], r["w"], r["h"]
    with openslide.OpenSlide(p) as s:
        vendor = s.properties.get("openslide.vendor", "?")
        afs = list(s.associated_images)
        print(f"[{i}] {Path(p).name} vendor={vendor}")
        print(f"    associated_files: {afs if afs else 'none'}")

        # true same physical region at L-1 / L / L+1: RGB + alpha
        # finer level (lower idx) -> coords x 2^n; coarser (higher idx) -> / 2^n
        same_phys = []
        for cand in (lv - 1, lv, lv + 1):
            if 0 <= cand < s.level_count:
                n = lv - cand
                if n > 0:    # cand finer -> multiply
                    fx, fy, fw, fh = x << n, y << n, w << n, h << n
                elif n < 0:  # cand coarser -> divide
                    fx, fy, fw, fh = x >> -n, y >> -n, w >> -n, h >> -n
                else:
                    fx, fy, fw, fh = x, y, w, h
                same_phys.append(_stats(s, cand, fx, fy, fw, fh))
        print(f"    same physical region: {' | '.join(same_phys)}")

        # overview at ~32x downsample, with markers
        best = s.get_best_level_for_downsample(32.0)
        W, H = s.level_dimensions[best]
        ov = s.read_region((0, 0), best, (W, H)).convert("RGB")
        ov2l0 = s.level_dimensions[0][0] / ov.width  # overview px -> level-0 px
        tbb = _bbox_nonblack(np.asarray(ov))

        draw = ImageDraw.Draw(ov)
        if tbb:
            x0, y0, x1, y1 = tbb
            draw.rectangle([x0, y0, x1, y1], outline="lime", width=2)
            print(
                f"    tissue_bbox_L0~({int(x0*ov2l0)},{int(y0*ov2l0)})-"
                f"({int(x1*ov2l0)},{int(y1*ov2l0)})"
            )
        lv_w, lv_h = s.level_dimensions[lv]
        kx, ky = ov.width / lv_w, ov.height / lv_h
        bx0, by0, bx1, by1 = int(x * kx), int(y * ky), int((x + w) * kx), int((y + h) * ky)
        draw.rectangle([bx0, by0, bx1, by1], outline="red", width=2)
        draw.line([bx0, by0, bx1, by1], fill="red", width=1)

        if ov.width > 1024:
            rr = 1024 / ov.width
            ov = ov.resize((1024, int(ov.height * rr)))

    ov.save(out / f"overview_{Path(p).stem}_{i}.png")
    print(f"    -> overview_{Path(p).stem}_{i}.png")


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
