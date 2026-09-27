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
    print(f"== 3. medoid crops ({len(medoids)} in jsonl, checking {n}) ==")
    try:
        import openslide
    except Exception as e:  # noqa: BLE001
        print(f"  skipped: openslide unavailable ({e})")
        return

    # spread across clusters
    by_c: dict[int, list[dict]] = {}
    for r in medoids:
        by_c.setdefault(r["cluster"], []).append(r)
    picks = []
    for c in sorted(by_c):
        picks.extend(by_c[c])
    idx = np.linspace(0, len(picks) - 1, min(n, len(picks))).round().astype(int)
    dark_nominal, dark_up, dark_dn, missing = 0, 0, 0, 0
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
                if lv >= s.level_count:
                    print(f"  c={r['cluster']}: stored level {lv} out of range "
                          f"(level_count={s.level_count})")
                    continue

                def mean_at(level: int, x: int, y: int, w: int, h: int) -> float | None:
                    W, H = s.level_dimensions[level]
                    if x + w > W or y + h > H:  # out of bounds -> informative miss
                        return None
                    img = s.read_region((x, y), level, (w, h)).convert("L")
                    return float(np.asarray(img).mean())

                m0 = mean_at(lv, r["x"], r["y"], r["w"], r["h"])
                line = f"  c={r['cluster']} {Path(p).name} ({r['x']},{r['y']}) "
                line += f"nominal L{lv} mean={m0:.0f}"
                if m0 is not None and m0 < 120:
                    dark_nominal += 1
                    up = mean_at(lv + 1, r["x"] * 2, r["y"] * 2, r["w"] * 2, r["h"] * 2)
                    dn = mean_at(max(0, lv - 1), r["x"] // 2, r["y"] // 2,
                                 max(1, r["w"] // 2), max(1, r["h"] // 2))
                    line += f"  | L{lv+1}: {up if up is None else round(up)} " \
                            f"L{max(0, lv-1)}: {dn if dn is None else round(dn)}"
                    if up is not None and up >= 120:
                        dark_up += 1
                    if dn is not None and dn >= 120:
                        dark_dn += 1
                print(line)
        except Exception as e:  # noqa: BLE001
            print(f"  c={r['cluster']} {Path(p).name}: ERROR {type(e).__name__}: {e}")
    print(
        f"  -> dark@nominal={dark_nominal}  (bright at L+1: {dark_up}, "
        f"bright at L-1: {dark_dn}, missing files: {missing})"
    )
    if dark_nominal and dark_up > dark_nominal / 2:
        print("  >>> bright one level DOWN in the pyramid: stored level is one too "
              "HIGH (coords are in the coarser level's space)")
    if dark_nominal and dark_dn > dark_nominal / 2:
        print("  >>> bright one level UP in the pyramid: stored level is one too LOW")
    if dark_nominal and not dark_up and not dark_dn:
        print("  >>> dark at every level: tiles really have no tissue there "
              "(tissue filter broken, or file != file tiling used)")


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
