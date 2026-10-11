"""Generate per-tile carcinoma scores from Virchow2 embeddings + classifier head.

Downloads the trained linear head from MLflow, runs it on tile embeddings,
and writes a CSV: slide_name, x, y, value.

Usage:
    python embeddings_to_heatmap.py \
        --embeddings "/path/to/Virchow2" \
        --slides slide_list.csv \
        --out heatmaps_tiles.csv \
        --min-value 0.5
"""

import argparse
import time
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

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
    # Look for the specific checkpoint: epoch=0-step=16816
    art = Path(artifact_dir)
    # MLflow lightning checkpoints are typically: checkpoints/epoch=0-step=16816.ckpt
    specific = list(art.glob("**/epoch=0-step=16816*"))
    if specific:
        ckpt = specific[0]
    else:
        # Fallback: any checkpoint
        candidates = list(art.glob("**/*.pt")) + list(art.glob("**/*.ckpt")) + list(art.glob("**/*.pth"))
        if not candidates:
            print(f"Available artifacts:")
            for f in sorted(art.rglob("*")):
                if f.is_file():
                    print(f"  {f.relative_to(art)}")
            raise SystemExit("No model checkpoint found in MLflow artifacts")
        ckpt = candidates[0]
        print(f"[warn] specific checkpoint not found, using: {ckpt}")
    print(f"Loading checkpoint: {ckpt}")

    import torch
    sd = torch.load(ckpt, map_location="cpu", weights_only=False)

    # Extract the linear head weights
    # The model has a backbone + a linear classifier head
    # Look for the last linear layer weights
    if isinstance(sd, dict):
        # Lightning checkpoint: sd["state_dict"]
        if "state_dict" in sd:
            sd = sd["state_dict"]
        # Find the classifier head keys
        head_keys = {k: v for k, v in sd.items() if "classifier" in k.lower() or "head" in k.lower() or "decode" in k.lower()}
        if not head_keys:
            # Fallback: find any linear layer with in_features=2560
            head_keys = {k: v for k, v in sd.items() if v.shape == (1, 2560) or v.shape == (2560,)}
        if not head_keys:
            print("Available keys:")
            for k in sd.keys():
                print(f"  {k}: {sd[k].shape}")
            raise SystemExit("Could not find classifier head weights")

    W = None
    b = None
    for k, v in head_keys.items():
        if v.shape == (1, 2560) or v.shape == (2560,):
            W = v.numpy().astype(np.float32).reshape(1, -1)
        elif v.shape == (1,) or v.shape == (1,):
            b = v.numpy().astype(np.float32).reshape(1)
    if W is None:
        raise SystemExit(f"Could not extract W from head keys: {list(head_keys.keys())}")
    if b is None:
        b = np.zeros(1, dtype=np.float32)
    return W, b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--embeddings", required=True, help="sharded dir with tiles/*.parquet")
    ap.add_argument("--slides", help="CSV with slide_name column (subset)")
    ap.add_argument("--out", default="heatmaps_tiles.csv")
    ap.add_argument("--min-value", type=float, default=0.5,
                    help="only write tiles with value >= this")
    ap.add_argument("--run", default=MODEL_RUN, help="MLflow run ID for model weights")
    args = ap.parse_args()

    # Load head weights
    t0 = time.monotonic()
    W, b = load_head_weights(args.run)
    print(f"Head loaded: W={W.shape}, b={b.shape} ({time.monotonic()-t0:.1f}s)")

    # Load slide_id -> slide_name mapping + target set
    id_to_name: dict[str, str] = {}
    if args.slides:
        slides_df = pd.read_csv(args.slides)
        if "slide_id" in slides_df.columns:
            for _, row in slides_df.iterrows():
                id_to_name[str(row["slide_id"])] = row["slide_name"]
            target_slides = set(slides_df["slide_id"].astype(str))
            print(f"Target: {len(target_slides)} slides (by slide_id)")
        else:
            target_slides = set(slides_df["slide_name"])
            print(f"Target: {len(target_slides)} slides (by name, no slide_id col)")
    else:
        target_slides = None

    # Index which parquet parts contain which slides
    tiles_dir = Path(args.embeddings) / "tiles" if (Path(args.embeddings) / "tiles").is_dir() else Path(args.embeddings)
    parts = sorted(tiles_dir.glob("*.parquet"))
    print(f"{len(parts)} tile parquet files")

    # Stream through parquets, compute scores, write CSV
    n_rows = 0
    t0 = time.monotonic()
    with open(args.out, "w", newline="") as f:
        f.write("slide_name,x,y,value\n")
        for pi, part in enumerate(parts):
            pf = pq.ParquetFile(str(part))
            for batch in pf.iter_batches(columns=["slide_id", "x", "y", "embedding"], batch_size=4096):
                sids = batch.column("slide_id").to_numpy(zero_copy_only=False)
                xs = batch.column("x").to_numpy(zero_copy_only=False).astype(np.int32)
                ys = batch.column("y").to_numpy(zero_copy_only=False).astype(np.int32)
                emb = batch.column("embedding").values.to_numpy(zero_copy_only=False)
                n = len(batch)
                emb = emb.reshape(n, -1).astype(np.float32)

                # L2 normalize (same as cluster_regions.py)
                norms = np.linalg.norm(emb, axis=1, keepdims=True)
                norms[norms == 0] = 1
                emb /= norms

                # Run head: sigmoid(W @ emb.T + b)
                logits = emb @ W.T + b  # (n, 1)
                scores = 1.0 / (1.0 + np.exp(-logits))  # (n, 1)
                scores = scores.ravel().astype(np.float32)

                # Get slide names
                names = []
                for sid in sids:
                    h = sid.hex() if isinstance(sid, (bytes, bytearray)) else str(sid)
                    names.append(id_to_name.get(h, h))

                if target_slides is not None:
                    mask_target = np.array([n in target_slides for n in names])
                else:
                    mask_target = np.ones(n, dtype=bool)

                mask = (scores >= args.min_value) & mask_target
                if mask.any():
                    pd.DataFrame({
                        "slide_name": [names[i] for i in np.where(mask)[0]],
                        "x": xs[mask],
                        "y": ys[mask],
                        "value": scores[mask],
                    }).to_csv(f, index=False, header=False)
                    n_rows += int(mask.sum())

            if (pi + 1) % 10 == 0:
                print(f"  {pi+1}/{len(parts)} ({time.monotonic()-t0:.0f}s, {n_rows} rows)")

    print(f"wrote {n_rows} rows -> {args.out} ({time.monotonic()-t0:.0f}s)")


if __name__ == "__main__":
    main()
