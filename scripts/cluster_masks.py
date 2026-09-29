"""Per-slide cluster masks (.tiff, xOpat-ready) from a clustering run.

Reads <out>/assignments.parquet (slide_id, x, y, cluster) + <out>/slides.parquet
(slide_id, path, level, tile_extent_x, tile_extent_y, mpp_x), paints each slide's
tiles into a uint8 canvas at the tiling level (value = cluster+1, 0 = background),
and saves with ratiopath.masks.write_big_tiff — the same pyramid-BigTIFF writer
ratiopath uses for xOpat overlays (512x512 tiles, DEFLATE, xres/yres from mpp).
Equal pixel values = tiles that matched into the same cluster.

Usage:
    uv run python scripts/cluster_masks.py --out clustering/k32 --dest masks
    uv run python scripts/cluster_masks.py --out clustering/k32 --dest masks --limit 5
    uv run python scripts/cluster_masks.py --out clustering/k32 --dest masks --slides <slide_id>
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyvips


def wsi_report(path, mpp_x: float):
    """{wh, level, mpp} at the WSI level whose mpp best matches mpp_x; None if unreadable."""
    try:
        import openslide

        with openslide.OpenSlide(str(path)) as s:
            base_mpp = s.mpp[0]
            best, best_diff, best_i = None, None, 0
            for i, (w, h) in enumerate(s.level_dimensions):
                m = base_mpp / (2**i)
                diff = abs(m - mpp_x) / mpp_x
                if best_diff is None or diff < best_diff:
                    best, best_diff, best_i = (w, h), diff, i
            base_wh, base_mpp0 = s.level_dimensions[0], s.mpp[0]
            return {"wh": best, "level": best_i, "mpp": base_mpp / (2**best_i),
                    "base_wh": base_wh, "base_mpp": base_mpp0}
    except Exception:
        return None


def paint(df_slide: pd.DataFrame, W: int, H: int, tw: int, th: int):
    """uint8 canvas (value=cluster+1) at the tiling level, painted onto (H, W)."""
    dtype = np.uint16 if int(df_slide["cluster"].max()) >= 255 else np.uint8
    x = df_slide["x"].to_numpy()
    y = df_slide["y"].to_numpy()
    c = df_slide["cluster"].to_numpy()
    canvas = np.zeros((H, W), dtype)
    x1 = np.clip(x, 0, W)
    y1 = np.clip(y, 0, H)
    x2 = np.minimum(x + tw, W)
    y2 = np.minimum(y + th, H)
    for xi, xj, yi, yj, v in zip(x1, x2, y1, y2, c, strict=True):
        if xj > xi and yj > yi:
            canvas[yi:yj, xi:xj] = int(v) + 1
    return canvas


def build_canvas(df_slide: pd.DataFrame, slide_row: pd.Series):
    """(canvas, W, H, source, report) at the tiling level."""
    tw, th = int(slide_row["tile_extent_x"]), int(slide_row["tile_extent_y"])
    x = df_slide["x"].to_numpy()
    y = df_slide["y"].to_numpy()
    W, H = int(x.max() + tw), int(y.max() + th)  # fallback: tile bounds
    report = wsi_report(slide_row["path"], float(slide_row["mpp_x"]))
    if report is not None:  # exact WSI canvas -> perfect xOpat alignment
        W, H, source = max(W, report["wh"][0]), max(H, report["wh"][1]), "wsi"
    else:
        source = "bounds"
    return paint(df_slide, W, H, tw, th), W, H, source, report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="clustering out dir (assignments+slides parquets)")
    ap.add_argument("--dest", required=True, help="dir to write <slide_id>.tiff into")
    ap.add_argument("--limit", type=int, default=0, help="only first N slides (0 = all)")
    ap.add_argument("--slides", nargs="*", help="only these slide_ids")
    ap.add_argument("--blend", action="store_true",
                    help="also write <name>.blend.png: mask blended onto the tissue "
                         "(ground truth for alignment, no xOpat involved)")
    ap.add_argument("--per-cluster", action="store_true",
                    help="write <dest>/cluster_<i>/<name>.tiff binary masks, one per cluster")
    ap.add_argument("--base-level", action="store_true",
                    help="force canvas to the WSI base (level-0) size — same size as the original image")
    ap.add_argument("--report-conf", default="",
                    help="config dir for the report tool: writes <dir>/reporter/tile_clusters.yaml; "
                         "run with: python -m report --config-dir <dir> reporter=tile_clusters "
                         "user=<your_name> mlflow=kubas_external")
    ap.add_argument("--slides-wsi-dir", default="",
                    help="WSI directory to put in the report config background")
    args = ap.parse_args()

    from ratiopath.masks.write_big_tiff import write_big_tiff

    from mask_preview import colorize  # same dir as this script; one palette everywhere

    a = pd.read_parquet(f"{args.out}/assignments.parquet")
    s = pd.read_parquet(f"{args.out}/slides.parquet")
    s = s.set_index("slide_id")
    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)

    slides = a["slide_id"].unique()
    if args.slides:
        slides = np.intersect1d(slides, args.slides)
    if args.limit:
        slides = slides[: args.limit]
    print(f"{len(slides)} slides to write -> {dest}")

    t0 = time.monotonic()
    used_names = set()
    failed = []
    align_rows = []
    for i, sid in enumerate(slides, 1):
        row = s.loc[sid]
        df = a[a["slide_id"] == sid]
        try:
            canvas, W, H, source, report = build_canvas(df, row)
            stem = Path(row["path"]).stem  # same name as the original WSI
            name, n = stem, 1
            while name in used_names:
                name = f"{stem}-{n}"
                n += 1
            used_names.add(name)
            if args.base_level and report is not None and (W, H) != tuple(report["base_wh"]):
                # same dimensions as the original WSI (level 0), streamed through vips —
                # the 2x upscale never materialises in RAM (a full-size PIL resize OOMs).
                f = report["base_wh"][0] / W  # == H0/H for isotropic WSIs
                to_base = lambda v: v.resize(f, kernel=pyvips.Kernel.NEAREST)
                mpp_x = float(report["base_mpp"])  # base-level mpp
                W, H = tuple(report["base_wh"])
            else:
                to_base = lambda v: v
                mpp_x = float(report["mpp"]) if report else float(row["mpp_x"])
            if args.per_cluster:
                k = int(df["cluster"].max())
                for ci in range(k + 1):
                    cmask = (canvas == ci + 1).astype(np.uint8)
                    cdir = dest / f"cluster_{ci:02d}"
                    cdir.mkdir(parents=True, exist_ok=True)
                    cimg = to_base(pyvips.Image.new_from_array(cmask))
                    write_big_tiff(cimg, cdir / f"{name}.tiff", mpp_x, mpp_x)
                    del cmask, cimg
            path = dest / f"{name}.tiff"
            img = to_base(pyvips.Image.new_from_array(canvas))
            write_big_tiff(img, path, mpp_x, mpp_x)
            if args.blend:
                # same-level raster composite: canvas is exactly the WSI level (source=wsi)
                try:
                    from PIL import Image as PImage

                    import openslide

                    with openslide.OpenSlide(str(row["path"])) as ws:
                        lvl = report["level"] if report else 0
                        wimg = ws.read_region((0, 0), lvl, ws.level_dimensions[lvl]).to_pil()
                    lab = PImage.fromarray(canvas).resize(
                        (wimg.width, wimg.height), PImage.NEAREST)
                    la = np.asarray(lab)
                    alpha = np.where(la > 0, 190, 0).astype(np.uint8)
                    col = PImage.fromarray(colorize(la, int(la.max())))
                    out = wimg.convert("RGBA")
                    out.paste(col, (0, 0), PImage.fromarray(alpha, "L"))
                    out.convert("RGB").save(dest / f"{name}.blend.png")
                    del wimg, out, col
                except Exception as e:  # preview only — never fail the mask itself
                    print(f"  {name}: blend preview failed: {e}")
            align_rows.append(
                (name, f"{W}x{H}", source,
                 f"{report['wh'][0]}x{report['wh'][1]}" if report else "UNREADABLE",
                 report["level"] if report else "", mpp_x, float(row["mpp_x"]))
            )
        except Exception as e:  # one bad WSI must not kill the batch
            failed.append((sid, str(e)))
            print(f"  [{i}/{len(slides)}] {sid[:12]}.. SKIPPED: {e}")
            continue
        del canvas, img
        print(f"  [{i}/{len(slides)}] {sid[:12]}.. {W}x{H} {source} "
              f"{len(df)} tiles {path.stat().st_size / 1e6:.0f} MB "
              f"({time.monotonic() - t0:.0f}s)")
    align = pd.DataFrame(
        align_rows, columns=["name", "canvas", "source", "wsi", "wsi_level", "mpp_written", "mpp_parquet"])
    for col in ("mpp_written", "mpp_parquet"):  # numeric even when 0 rows (empty -> object)
        align[col] = pd.to_numeric(align[col], errors="coerce")
    align["mpp_mismatch_pct"] = (
        100 * (align["mpp_written"] - align["mpp_parquet"]).abs() / align["mpp_parquet"]).round(3)
    align.to_csv(dest / "alignment.csv", index=False)
    print(f"alignment report: {dest / 'alignment.csv'}")
    print(align.to_string(index=False))

    if args.report_conf:
        k_total = int(a["cluster"].max())
        wsi_dir = str(Path(args.slides_wsi_dir) if args.slides_wsi_dir else "WSI_DIR")
        lines = [
            "defaults:",
            "  - default",
            "  - save: local",
            "",
            "title: Tile morphology clusters",
            "background:",
            "  _target_: report.masks.BasicImageRetriever",
            f"  source_dir: {wsi_dir}",
            '  globs: ["*.svs", "*.mrxs", "*.tif", "*.tiff"]',
            "  layer_name: WSI background",
            "mask_retrievers:",
        ]
        if args.per_cluster:
            for ci in range(k_total + 1):
                lines += [
                    f"  - _target_: report.masks.BasicImageRetriever",
                    f"    source_dir: {dest}/cluster_{ci:02d}",
                    '    globs: ["*.tiff"]',
                    f'    layer_name: Cluster {ci}',
                ]
        else:  # one mask per slide carrying all clusters
            lines += [
                "  - _target_: report.masks.BasicImageRetriever",
                f"    source_dir: {dest}",
                '    globs: ["*.tiff"]',
                "    layer_name: Cluster labels (0=bg, value=cluster+1)",
            ]
        lines += ["save:", "  output_path: report.html"]
        conf_path = Path(args.report_conf) / "reporter" / "tile_clusters.yaml"
        conf_path.parent.mkdir(parents=True, exist_ok=True)
        conf_path.write_text("\n".join(lines) + "\n")
        print(f"report config: {conf_path}")
        print(f"run: python -m report --config-dir {Path(args.report_conf)} "
              f"reporter=tile_clusters user=<your_name> mlflow=kubas_external")

    print(f"done: {len(slides) - len(failed)} written in {time.monotonic() - t0:.0f}s")
    if failed:
        print(f"WARNING: {len(failed)} slides failed:")
        for sid, err in failed[:10]:
            print(f"  {sid[:12]}.. {err}")
        sys.exit(1)


if __name__ == "__main__":
    main()
