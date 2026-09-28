"""Read-only: why are the medoid crops black?

For each of the first N medoids, reads the SAME physical region at three levels
around the stored (level-1) crop point and discriminates the two causes:
  A: tile sits on background        -> L0 tissue, L1 empty, L2 empty
     (tiling/mask placed tiles on background gaps; fix = same-level mask)
  B: stored level is one too fine  -> L0 tissue, L1 empty, L2 tissue
     (data really at L0; fix = read at L0, coords x2, no re-tile)

No files written. Run on the WSI node:
    uv run python scripts/level_probe.py --medoids <out>/medoids.jsonl --n 6
"""

import argparse
import json
from pathlib import Path

import numpy as np
import openslide


def _rgb(s, level, x, y, w, h) -> str:
    W, H = s.level_dimensions[level]
    if x < 0 or y < 0 or x >= W or y >= H:
        return f"L{level} OOB"
    w, h = min(w, W - x), min(h, H - y)
    if w <= 0 or h <= 0:
        return f"L{level} OOB"
    a = np.asarray(s.read_region((x, y), level, (w, h)).convert("RGB"))
    return f"L{level} rgb{float(a.mean()):.0f}"


def probe_medoid(r: dict, i: int) -> None:
    """At the medoid's stored (level-1) point, read the SAME physical region at
    L0 (x2), L1 (stored), L2 (/2). Discriminates:
      - A (tile on background):  L0 tissue, L1 empty, L2 empty
      - B (level 1 too fine):    L0 tissue, L1 empty, L2 tissue
    """
    p, lv = r["slide_path"], int(r["level"])
    x, y, w, h = int(r["x"]), int(r["y"]), int(r["w"]), int(r["h"])
    name = Path(p).name
    if not Path(p).exists():
        print(f"[{i}] MISSING {p}")
        return
    with openslide.OpenSlide(p) as s:
        # same physical region: level k -> coords /2^(lv-k)
        l0 = _rgb(s, 0, x << lv, y << lv, w << lv, h << lv)
        l1 = _rgb(s, lv, x, y, w, h)
        l2 = _rgb(s, lv + 1, x >> 1, y >> 1, max(1, w >> 1), max(1, h >> 1))
        def rgbval(t: str) -> float:
            return float(t.split("rgb")[1]) if "rgb" in t else -1.0
        l0_tissue, l2_tissue = rgbval(l0) >= 20, rgbval(l2) >= 20
        verdict = "?"
        if l0_tissue and l2_tissue:
            verdict = "B: level-1 too fine -> read at L0 (coords x2), no re-tile"
        elif l0_tissue and not l2_tissue:
            verdict = "A: tile on background -> re-tile with same-level mask"
        print(f"[{i}] {name} stored level={lv} pt=({x},{y}) size={w}x{h}")
        print(f"      same physical region: {l0} | {l1} | {l2}")
        print(f"      -> {verdict}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--medoids", required=True)
    ap.add_argument("--n", type=int, default=6)
    args = ap.parse_args()
    with open(args.medoids) as f:
        rows = [json.loads(l) for l in f if l.strip()]
    for i, r in enumerate(rows[: args.n]):
        probe_medoid(r, i)


if __name__ == "__main__":
    main()
