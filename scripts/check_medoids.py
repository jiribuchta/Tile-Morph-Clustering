"""Diagnose black medoid tiles. Run on a node with the WSI + parquet mounts.

Checks, for a sharded folder (tiles/*.parquet + slides/slides.parquet) and an
optional medoids.jsonl from a clustering run:

1. tissue scale   - is tissue_roi_percentage in [0,1] and does >= 0.5 select tissue?
2. slide geometry - does slides.parquet level/mpp match the WSI's actual levels?
3. medoid crops   - are the crops black at (x, y, level, w, h)? if so, are they
                    bright at level+1 / level-1 (=> level mismatch) or dark
                    everywhere (=> tile really has no tissue / wrong file)?

Usage:
    uv run python scripts/check_medoids.py \
        --sharded "/mnt/projects/.../MMCI B20-24 Train/Virchow2" \
        [--medoids /path/to/medoids.jsonl] [--slides N] [--medoids-n M]
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


def check_tissue_scale(sharded: Path, parts: int = 3) -> None:
    print("== 1. tissue_roi_percentage scale (tile parquets) ==")
    tile_dir = sharded / "tiles" if (sharded / "tiles").is_dir() else sharded
    files = sorted(tile_dir.glob("*.parquet"))[:parts]
    vals = []
    for f in files:
        t = pq.read_table(f, columns=["tissue_roi_percentage"]).column(
            "tissue_roi_percentage"
        ).to_pylist()
        vals.extend(t)
    v = np.asarray(vals, dtype=float)
    print(
        f"  {len(files)} parts, {len(v)} rows: min={v.min():.4f} max={v.max():.4f} "
        f"p50={np.percentile(v, 50):.4f}  frac<0.5={float((v < 0.5).mean()):.3f} "
        f"frac>=0.5={float((v >= 0.5).mean()):.3f} frac>=0.99={float((v >= 0.99).mean()):.3f}"
    )
    if v.max() > 2:
        print("  >>> MAX > 2: column looks like 0-100 scale! filter min_tissue=0.5 "
              "keeps everything, including empty tiles")


def check_slide_geometry(slides: pd.DataFrame, sharded: Path, n: int = 3) -> None:
    print(f"== 2. slide geometry (slides.parquet vs {n} WSIs) ==")
    print(
        f"  slides.parquet: level={slides['level'].unique().tolist()} "
        f"mpp_x={slides['mpp_x'].round(4).unique()[:5]} "
        f"tile_extent_x={slides['tile_extent_x'].unique().tolist()}"
    )
    try:
        import openslide
    except Exception as e:  # noqa: BLE001
        print(f"  skipped: openslide unavailable ({e})")
        return

    def mpp0(s):
        raw = s.properties.get("openslide.mpp")
        try:
            return tuple(float(t) for t in raw.split(","))
        except Exception:
            return None

    for i, r in slides.head(n).iterrows():
        p = r["path"]
        if not Path(p).exists():
            print(f"  {p}: MISSING FILE")
            continue
        with openslide.OpenSlide(p) as s:
            base = mpp0(s)
            dims = []
            for lv in range(s.level_count):
                W, H = s.level_dimensions[lv]
                m = f" mpp={base[0] * s.level_downsamples[lv][0]:.4f}" if base else ""
                dims.append(f"L{lv}: {W}x{H}{m}")
            print(f"  {Path(p).name}: {' | '.join(dims)}")
            lv = int(r["level"])
            W, H = s.level_dimensions[lv]
            lv_mpp = base[0] * s.level_downsamples[lv][0] if base else float("nan")
            print(
                f"    stored level={lv} (dim {W}x{H}), mpp_x stored={r['mpp_x']:.4f} "
                f"vs WSI~{lv_mpp:.4f}"
            )


def check_medoid_crops(medoids: list[dict], n: int = 12) -> None:
    """Read each medoid exactly as the working pattern: read_region((x,y), level,
    (w,h)) in level-space coords; empty == alpha.max()==0. Also reads the SAME
    physical region at level 0 (coords x 2^level) so we can tell 'tile is on
    background' from 'stored level is the wrong one'.
    """
    print(f"== 3. medoid crops ({len(medoids)} in jsonl, checking {n}) ==")
    try:
        import openslide
    except Exception as e:  # noqa: BLE001
        print(f"  skipped: openslide unavailable ({e})")
        return

    by_c: dict[int, list[dict]] = {}
    for r in medoids:
        by_c.setdefault(r["cluster"], []).append(r)
    picks = [r for c in sorted(by_c) for r in by_c[c]]
    idx = np.linspace(0, len(picks) - 1, min(n, len(picks))).round().astype(int)

    empty_nom, ok_nom = 0, 0          # stored-level read empty / has data
    ok_l0 = 0                          # same region non-empty at L0
    missing = 0
    for i in idx:
        r = picks[i]
        p = r["slide_path"]
        if not Path(p).exists():
            missing += 1
            print(f"  c={r['cluster']} {Path(p).name}: MISSING FILE")
            continue
        try:
            with openslide.OpenSlide(p) as s:
                lv = int(r["level"])
                x, y, w, h = int(r["x"]), int(r["y"]), int(r["w"]), int(r["h"])
                if lv >= s.level_count:
                    print(f"  c={r['cluster']}: stored level {lv} OOR "
                          f"(level_count={s.level_count})")
                    continue

                def stats(level: int, px: int, py: int, pw: int, ph: int):
                    """(alpha_max, rgb_mean) or (None, None) if out of bounds."""
                    W, H = s.level_dimensions[level]
                    if px < 0 or py < 0 or px >= W or py >= H:
                        return None, None
                    pw, ph = min(pw, W - px), min(ph, H - py)
                    if pw <= 0 or ph <= 0:
                        return None, None
                    a = np.asarray(s.read_region((px, py), level, (pw, ph)))
                    amax = int(a[..., 3].max()) if a.shape[-1] == 4 else 255
                    return amax, float(a[..., :3].mean())

                a_nom, m_nom = stats(lv, x, y, w, h)
                a_l0, m_l0 = stats(0, x << lv, y << lv, w << lv, h << lv)  # same region
                if a_nom is None:
                    print(f"  c={r['cluster']} {Path(p).name}: stored level OOB")
                    continue
                tag_nom = "EMPTY" if a_nom == 0 else f"a{a_nom} rgb{m_nom:.0f}"
                tag_l0 = ("OOB" if a_l0 is None
                          else ("EMPTY" if a_l0 == 0 else f"a{a_l0} rgb{m_l0:.0f}"))
                print(f"  c={r['cluster']} {Path(p).name} ({x},{y}) L{lv} {w}x{h}")
                print(f"      stored-level read : {tag_nom}")
                print(f"      L0 same region    : {tag_l0}")
                if a_nom == 0:
                    empty_nom += 1
                else:
                    ok_nom += 1
                if a_l0 is not None and a_l0 > 0:
                    ok_l0 += 1
        except Exception as e:  # noqa: BLE001
            print(f"  c={r['cluster']} {Path(p).name}: ERROR {type(e).__name__}: {e}")
    print(f"  -> stored-level: empty={empty_nom}  has-data={ok_nom}  "
          f"L0-same-region has-data={ok_l0}  missing={missing}")
    if empty_nom:
        if ok_l0 >= empty_nom / 2:
            print("  >>> stored level empty but SAME REGION HAS DATA AT L0: the read "
                  "level is the problem, not the coords. Read the tile at level 0 "
                  "(montage.py already does: coords x 2^level). No re-tile needed.")
        else:
            print("  >>> empty at the stored level AND empty at L0 same region: the "
                  "tile coords sit on background (tiling/mask placed grid tiles off-"
                  "tissue). Durable fix = re-tile with a mask at the grid's level.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sharded", required=True,
                    help="sharded folder with tiles/*.parquet and slides/slides.parquet")
    ap.add_argument("--medoids", default=None, help="medoids.jsonl from a clustering run")
    ap.add_argument("--slides-n", type=int, default=3)
    ap.add_argument("--medoids-n", type=int, default=12)
    args = ap.parse_args()

    sharded = Path(args.sharded)
    slides_path = sharded / "slides" / "slides.parquet"
    if not slides_path.exists():
        slides_path = sharded / "slides.parquet"
    slides = pd.read_parquet(slides_path)

    check_tissue_scale(sharded)
    check_slide_geometry(slides, sharded, args.slides_n)
    if args.medoids:
        with open(args.medoids) as f:
            check_medoid_crops(
                [json.loads(l) for l in f if l.strip()], args.medoids_n
            )


if __name__ == "__main__":
    main()
