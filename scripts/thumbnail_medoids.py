"""Debug black medoid crops: dump raw crops + slide frame metadata + overview.

Writes a small contact sheet per medoid (``<out>/medoid_<cluster>_<i>.png``) and
one overview per slide (``<out>/overview_<slide>.png``), and prints the slide's
OpenSlide properties / associated files so we can see whether the ``.mrxs``
is multiframe and which frame ``read_region`` is reading.

Run on the node with the WSI mount:
    uv run python scripts/thumbnail_medoids.py \
        --medoids <out>/medoids.jsonl --out /tmp/medoid_dbg --max 6
"""

import argparse
import json
from pathlib import Path

import numpy as np
import openslide
from PIL import Image


def _mean(img: Image.Image) -> float:
    return float(np.asarray(img.convert("L")).mean())


def slide_overview(path: str, out: Path) -> None:
    """Downsample the whole slide to <=1024px wide and save it."""
    with openslide.OpenSlide(path) as s:
        best = s.get_best_level_for_downsample(1.0 / 32)
        w0, h0 = s.level_dimensions[best]
        img = s.read_region((0, 0), best, (w0, h0))
        if img.width > 1024:
            r = 1024 / img.width
            img = img.resize((1024, int(img.height * r)))
    img.save(out / f"overview_{Path(path).stem}.png")


def medoid_sheet(r: dict, out: Path, tag: str) -> None:
    """One sheet: raw crop at nominal level, plus L-1 / L+1, plus 1 overview tile."""
    p = r["slide_path"]
    lv, x, y, w, h = r["level"], r["x"], r["y"], r["w"], r["h"]
    cells, notes = [], []
    with openslide.OpenSlide(p) as s:
        nlevels = s.level_count

        def crop(level: int, cx: int, cy: int, cw: int, ch: int) -> Image.Image | None:
            if level < 0 or level >= nlevels:
                return None
            W, H = s.level_dimensions[level]
            if cx >= W or cy >= H:
                return None
            cw = min(cw, W - cx)
            ch = min(ch, H - cy)
            return s.read_region((cx, cy), level, (cw, ch))

        nominal = crop(lv, x, y, w, h)
        if nominal is not None:
            nominal = nominal.resize((224, 224))
            cells.append(nominal)
            notes.append(f"L{lv} mean={_mean(nominal):.0f}")
        for cand, lab in [
            (crop(lv - 1, x // 2, y // 2, w // 2, h // 2), f"L{lv-1}"),
            (crop(lv + 1, x * 2, y * 2, w * 2, h * 2), f"L{lv+1}"),
        ]:
            if cand is not None:
                cells.append(cand.resize((224, 224)))
                notes.append(f"{lab} mean={_mean(cand):.0f}")

    canvas = Image.new("RGB", (230 * len(cells) + 4, 250), "white")
    for i, c in enumerate(cells):
        canvas.paste(c, (i * 230 + 2, 20))
    from PIL import ImageDraw

    ImageDraw.Draw(canvas).text((2, 2), " ".join(notes), fill="black")
    canvas.save(out / f"medoid_{tag}.png")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--medoids", required=True)
    ap.add_argument("--out", default="/tmp/medoid_dbg")
    ap.add_argument("--max", type=int, default=6)
    ap.add_argument("--overview", type=int, default=3,
                    help="how many distinct slides to dump a full overview for")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    with open(args.medoids) as f:
        rows = [json.loads(l) for l in f if l.strip()]
    rows = rows[: args.max]

    seen = set()
    for i, r in enumerate(rows):
        p = Path(r["slide_path"])
        print(f"[{i}] c={r['cluster']} {p.name} level={r['level']} "
              f"({r['x']},{r['y']}) {r['w']}x{r['h']}")
        if not p.exists():
            print("    MISSING FILE")
            continue
        with openslide.OpenSlide(p) as s:
            w0, h0 = s.level_dimensions[0]
            print(f"    level_count={s.level_count}  L0 size={w0}x{h0}")
            props = {k: v for k, v in s.properties.items()
                     if any(t in k.lower() for t in ("mpp", "vendor", "frame", "type"))}
            for k, v in props.items():
                print(f"    {k} = {v}")
            afs = list(s.associated_files)
            if afs:
                print(f"    associated_files: {afs[:8]}")
        medoid_sheet(r, out, tag=f"{r['cluster']}_{i}")
        if p.name not in seen and len(seen) < args.overview:
            seen.add(p.name)
            slide_overview(p, out)
            print(f"    -> overview_{p.stem}.png")
    print(f"\nwrote sheets to {out}")


if __name__ == "__main__":
    main()
