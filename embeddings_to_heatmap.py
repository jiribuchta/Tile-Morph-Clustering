"""Generate per-tile carcinoma scores from Virchow2 embeddings + classifier head.

Downloads the trained linear head from MLflow, runs it on tile embeddings,
and writes one .parquet per slide: slide_name, x, y, value.
Resumable: existing per-slide parquets are skipped on rerun.

Usage (per-slide .pt files — preferred):
    python embeddings_to_heatmap.py \
        --pt-dir "/mnt/.../tile_embeddings/virchow2/MMCI B20-24 Val" \
        --slides slide_list.csv \
        --out heatmaps_tiles \
        --min-value 0.5

Usage (sharded parquets):
    python embeddings_to_heatmap.py \
        --embeddings "/path/to/Virchow2" \
        --slides slide_list.csv \
        --out heatmaps_tiles \
        --min-value 0.5
"""

import argparse
import time
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch

TRACKING_URI = "http://mlflow.rationai-mlflow:5000"
# Virchow2 (Frozen) selected model from BRC-A-29
MODEL_RUN = "b760612d87884b4a866fea9f460ed214"


def load_head_weights(run_id: str) -> tuple[np.ndarray, np.ndarray]:
    """Download classifier head (linear layer) weights from MLflow.

    Returns (W: (1, 2560), b: (1,)) as float32.
    """
    mlflow.set_tracking_uri(TRACKING_URI)
    artifact_dir = mlflow.artifacts.download_artifacts(
        f"mlflow-artifacts:/61/{run_id}/artifacts"
    )
    art = Path(artifact_dir)
    # Look for the specific checkpoint: epoch=0-step=16816
    specific_dirs = list(art.glob("**/epoch=0-step=16816"))
    specific_files = list(art.glob("**/epoch=0-step=16816.*"))
    if specific_files:
        ckpt = specific_files[0]
    elif specific_dirs:
        inner = list(specific_dirs[0].glob("*.ckpt")) + list(specific_dirs[0].glob("*.pt"))
        if not inner:
            print(f"Contents of {specific_dirs[0]}:")
            for f in specific_dirs[0].iterdir():
                print(f"  {f.name}")
            raise SystemExit("No checkpoint file inside directory")
        ckpt = inner[0]
    else:
        candidates = [f for f in art.rglob("*") if f.suffix in (".ckpt", ".pt", ".pth")]
        if not candidates:
            print(f"Available artifacts:")
            for f in sorted(art.rglob("*")):
                if f.is_file():
                    print(f"  {f.relative_to(art)}")
            raise SystemExit("No model checkpoint found in MLflow artifacts")
        ckpt = candidates[0]
        print(f"[warn] specific checkpoint not found, using: {ckpt}")
    print(f"Loading checkpoint: {ckpt}")

    sd = torch.load(ckpt, map_location="cpu", weights_only=False)

    if isinstance(sd, dict):
        if "state_dict" in sd:
            sd = sd["state_dict"]
        head_keys = {k: v for k, v in sd.items() if "classifier" in k.lower() or "head" in k.lower() or "decode" in k.lower()}
        if not head_keys:
            head_keys = {k: v for k, v in sd.items() if v.shape == (1, 2560) or v.shape == (2560,)}
        if not head_keys:
            print("Available keys:")
            for k in sd.keys():
                print(f"  {k}: {sd[k].shape}")
            raise SystemExit("Could not find classifier head weights")
    else:
        raise SystemExit(f"Unexpected checkpoint format: {type(sd)}")

    W = None
    b = None
    for k, v in head_keys.items():
        if v.shape == (1, 2560) or v.shape == (2560,):
            W = v.numpy().astype(np.float32).reshape(1, -1)
        elif v.shape == (1,):
            b = v.numpy().astype(np.float32).reshape(1)
    if W is None:
        raise SystemExit(f"Could not extract W from head keys: {list(head_keys.keys())}")
    if b is None:
        b = np.zeros(1, dtype=np.float32)
    return W, b


def process_pt_files(pt_dir: Path, target_names: set | None, W: np.ndarray, b: np.ndarray,
                     min_value: float, out: str):
    """Process per-slide .pt files."""
    pt_files = sorted(pt_dir.glob("*.pt"))
    print(f"{len(pt_files)} .pt files in {pt_dir}")

    if target_names is not None:
        pt_files = [p for p in pt_files if p.stem in target_names]
        print(f"  {len(pt_files)} match target slides")

    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    n_rows = 0
    n_tiles_total = 0
    n_skipped = 0
    t0 = time.monotonic()
    for pi, pt_file in enumerate(pt_files):
        slide_name = pt_file.stem
        out_path = out_dir / f"{slide_name}.parquet"
        if out_path.exists():
            n_skipped += 1
            continue  # resume: slide already done
        data = torch.load(pt_file, map_location="cpu", weights_only=True)

        # Handle different formats
        if isinstance(data, dict):
            emb = None
            xs = None
            ys = None
            for k in ("embeddings", "embedding", "emb", "features"):
                if k in data:
                    emb = data[k].numpy().astype(np.float32)
                    break
            for k in ("x", "xs", "x_coords"):
                if k in data:
                    xs = data[k].numpy().astype(np.int32)
                    break
            for k in ("y", "ys", "y_coords"):
                if k in data:
                    ys = data[k].numpy().astype(np.int32)
                    break
            if emb is None:
                if pi == 0:
                    print(f"  [debug] keys in {pt_file.name}: {list(data.keys())}")
                    for k, v in data.items():
                        print(f"    {k}: {type(v).__name__}", getattr(v, 'shape', ''))
                raise SystemExit(f"Cannot find embeddings in {pt_file.name}")
            if xs is None or ys is None:
                n = emb.shape[0]
                side = int(n ** 0.5)
                xs = np.tile(np.arange(side), n // side).astype(np.int32)
                ys = np.repeat(np.arange(n // side), side).astype(np.int32)
                print(f"  [warn] {pt_file.name}: no coords, generated {side}x{side} grid")
        else:
            emb = data.numpy().astype(np.float32)
            n = emb.shape[0]
            side = int(n ** 0.5)
            xs = np.tile(np.arange(side), n // side).astype(np.int32)
            ys = np.repeat(np.arange(n // side), side).astype(np.int32)
            print(f"  [warn] {pt_file.name}: tensor only, generated {side}x{side} grid")

        n = emb.shape[0]
        n_tiles_total += n

        # Run head: sigmoid(W @ emb + b) — no L2 norm
        logits = emb @ W.T + b
        scores = (1.0 / (1.0 + np.exp(-logits))).ravel().astype(np.float32)

        if pi == 0:
            print(f"  [debug] {slide_name}: {n} tiles, score range [{scores.min():.4f}, {scores.max():.4f}], mean {scores.mean():.4f}")

        mask = scores >= min_value
        df = pd.DataFrame({
            "slide_name": slide_name,
            "x": xs[mask],
            "y": ys[mask],
            "value": scores[mask],
        })
        df.to_parquet(out_path)  # written even if empty: marks slide as done
        n_rows += len(df)
        if (pi + 1) % 25 == 0 or pi + 1 == len(pt_files):
            print(f"  {pi+1}/{len(pt_files)} slides, {n_rows} carcinoma rows ({time.monotonic()-t0:.0f}s)", flush=True)

    print(f"total tiles: {n_tiles_total}")
    print(f"wrote {n_rows} rows -> {out_dir} ({time.monotonic()-t0:.0f}s, {n_skipped} skipped)")


def process_parquets(embed_dir: str, target_slides: set | None, id_to_name: dict,
                     W: np.ndarray, b: np.ndarray, min_value: float, out: str):
    """Process sharded parquet files."""
    tiles_dir = Path(embed_dir) / "tiles" if (Path(embed_dir) / "tiles").is_dir() else Path(embed_dir)
    all_parts = sorted(tiles_dir.glob("*.parquet"))
    print(f"{len(all_parts)} tile parquet files")

    if target_slides is not None:
        part_to_ids: dict[str, set] = {}
        t0 = time.monotonic()
        for p in all_parts:
            pf = pq.ParquetFile(str(p))
            for batch in pf.iter_batches(columns=["slide_id"], batch_size=65536):
                for sid in batch.column("slide_id").to_numpy(zero_copy_only=False):
                    h = sid.hex() if isinstance(sid, (bytes, bytearray)) else str(sid)
                    if h in target_slides:
                        part_to_ids.setdefault(str(p), set()).add(h)
        target_parts = [p for p in all_parts if str(p) in part_to_ids]
        found_ids = set().union(*part_to_ids.values()) if part_to_ids else set()
        missing = target_slides - found_ids
        if missing:
            print(f"  [warn] {len(missing)} target slides not found, e.g. {list(missing)[:5]}")
        print(f"  scan done: {len(target_parts)} parts contain {len(found_ids)} target slides ({time.monotonic()-t0:.1f}s)")
    else:
        target_parts = all_parts

    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    n_rows = 0
    n_target_total = 0
    t0 = time.monotonic()
    for pi, part in enumerate(target_parts):
        out_path = out_dir / f"part_{pi:06d}.parquet"
        if out_path.exists():
            continue  # resume: part already done
        pf = pq.ParquetFile(str(part))
        part_rows = []
        for batch in pf.iter_batches(columns=["slide_id", "x", "y", "embedding"], batch_size=4096):
            sids = batch.column("slide_id").to_numpy(zero_copy_only=False)
            xs = batch.column("x").to_numpy(zero_copy_only=False).astype(np.int32)
            ys = batch.column("y").to_numpy(zero_copy_only=False).astype(np.int32)
            emb = batch.column("embedding").values.to_numpy(zero_copy_only=False)
            n = len(batch)
            emb = emb.reshape(n, -1).astype(np.float32)

            logits = emb @ W.T + b
            scores = (1.0 / (1.0 + np.exp(-logits))).ravel().astype(np.float32)

            names = []
            for sid in sids:
                h = sid.hex() if isinstance(sid, (bytes, bytearray)) else str(sid)
                names.append(id_to_name.get(h, h))

            if target_slides is not None:
                mask_target = np.array([nm in target_slides for nm in names])
            else:
                mask_target = np.ones(n, dtype=bool)

            mask = (scores >= min_value) & mask_target
            n_target = int(mask_target.sum())
            n_target_total += n_target
            if n_target > 0 and not hasattr(process_parquets, '_debugged'):
                process_parquets._debugged = True
                print(f"  [debug] first batch with target: {n_target}/{n} tiles")
                print(f"  [debug] target score range [{scores[mask_target].min():.4f}, {scores[mask_target].max():.4f}], mean {scores[mask_target].mean():.4f}")
            if mask.any():
                part_rows.append(pd.DataFrame({
                    "slide_name": [names[i] for i in np.where(mask)[0]],
                    "x": xs[mask],
                    "y": ys[mask],
                    "value": scores[mask],
                }))
                n_rows += int(mask.sum())

        df = pd.concat(part_rows, ignore_index=True) if part_rows else pd.DataFrame(columns=["slide_name", "x", "y", "value"])
        df.to_parquet(out_path)
        if (pi + 1) % 10 == 0 or pi + 1 == len(target_parts):
            print(f"  {pi+1}/{len(target_parts)} parts, {n_rows} rows ({time.monotonic()-t0:.0f}s)", flush=True)

    if target_slides is not None:
        print(f"total target tiles: {n_target_total}")
    print(f"wrote {n_rows} rows -> {out_dir} ({time.monotonic()-t0:.0f}s)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--embeddings", help="sharded dir with tiles/*.parquet")
    ap.add_argument("--pt-dir", help="dir with per-slide .pt files (simpler)")
    ap.add_argument("--slides", help="CSV with slide_name/slide_id column (subset)")
    ap.add_argument("--out", default="heatmaps_tiles",
                    help="output dir: one .parquet per slide/part (resumable)")
    ap.add_argument("--min-value", type=float, default=0.5,
                    help="only write tiles with value >= this")
    ap.add_argument("--run", default=MODEL_RUN, help="MLflow run ID for model weights")
    args = ap.parse_args()

    if not args.embeddings and not args.pt_dir:
        raise SystemExit("--embeddings or --pt-dir required")

    # Load head weights
    t0 = time.monotonic()
    W, b = load_head_weights(args.run)
    print(f"Head loaded: W={W.shape}, b={b.shape} ({time.monotonic()-t0:.1f}s)")

    # Load slide list
    id_to_name: dict[str, str] = {}
    target_slides = None
    target_names = None
    if args.slides:
        slides_df = pd.read_csv(args.slides)
        if "slide_id" in slides_df.columns:
            for _, row in slides_df.iterrows():
                id_to_name[str(row["slide_id"])] = row["slide_name"]
            target_slides = set(slides_df["slide_id"].astype(str))
            target_names = set(slides_df["slide_name"])
            print(f"Target: {len(target_slides)} slides")
        else:
            target_slides = set(slides_df["slide_name"])
            target_names = target_slides
            print(f"Target: {len(target_slides)} slides (by name)")

    # Dispatch
    if args.pt_dir:
        process_pt_files(Path(args.pt_dir), target_names, W, b, args.min_value, args.out)
    else:
        process_parquets(args.embeddings, target_slides, id_to_name, W, b, args.min_value, args.out)


if __name__ == "__main__":
    main()
