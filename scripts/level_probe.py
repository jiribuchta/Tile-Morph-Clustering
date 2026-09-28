"""Read-only: which pyramid level actually carries pixels in these .mrxs?

For each of the first N slides, finds the tissue centroid from the coarsest
level, then reads a 64x64 patch at that SAME physical point (correctly scaled)
on every level and prints RGB + alpha.

Separates three hypotheses:
  - full-res levels genuinely empty  -> L0/L1 rgb0 while coarse levels show tissue
  - wrong level order                -> pixels appear at an unexpectedly high level
  - wrong file / missing data       -> coarse level itself has no tissue

No files written. Run on the WSI node:
    uv run python scripts/level_probe.py --from-medoids <out>/medoids.jsonl --n 3
    (or --slides <a.mrxs> <b.mrxs> ...)
"""

import argparse
import json
from pathlib import Path

import numpy as np
import openslide


def probe(path: str) -> None:
    name = Path(path).name
    with openslide.OpenSlide(path) as s:
        lv = s.level_count - 1
        W, H = s.level_dimensions[lv]
        coarse = np.asarray(s.read_region((0, 0), lv, (W, H)).convert("RGB"))
        g = coarse.max(axis=-1)
        ys, xs = np.where(g > 16)
        if not len(xs):
            print(f"== {name}  NO tissue found at coarsest level?!")
            return
        cx = xs.mean() * (s.level_dimensions[0][0] / W)
        cy = ys.mean() * (s.level_dimensions[0][1] / H)
        print(f"== {name}  tissue centroid L0~({int(cx)},{int(cy)})  px@coarse={len(xs)}")
        for lv in range(s.level_count):
            W, H = s.level_dimensions[lv]
            px = min(int(cx >> lv), max(0, W - 64))
            py = min(int(cy >> lv), max(0, H - 64))
            a = np.asarray(s.read_region((px, py), lv, (64, 64)))
            al = f"a{a[..., 3].min()}-{a[..., 3].max()}" if a.shape[-1] == 4 else "no-a"
            print(f"  L{lv} {W}x{H}  patch@({px},{py}) "
                  f"rgb={float(a[..., :3].mean()):.0f} {al}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--slides", nargs="*", default=[])
    ap.add_argument("--from-medoids", default=None)
    ap.add_argument("--n", type=int, default=3)
    args = ap.parse_args()
    slides = list(args.slides)
    if args.from_medoids:
        with open(args.from_medoids) as f:
            seen, rows = set(), []
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                if r["slide_path"] not in seen:
                    seen.add(r["slide_path"])
                    rows.append(r["slide_path"])
        slides = slides + rows
    for p in slides[: args.n]:
        probe(p)


if __name__ == "__main__":
    main()
