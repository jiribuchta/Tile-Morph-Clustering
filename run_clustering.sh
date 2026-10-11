#!/bin/bash
# Job: cluster tiles with KMeans
# Output: /mnt/projects/breast_cancer/tile_morph_clustering/tile_clustering/

set -e

PT_DIR="/mnt/projects/breast_cancer/bc/tile_embeddings/virchow2/MMCI B20-24 Val"
TILES="/mnt/projects/breast_cancer/tile_morph_clustering/test_heatmaps.csv"
MORPH="/mnt/projects/breast_cancer/tile_morph_clustering/slide_list_balanced.csv"
OUT="/mnt/projects/breast_cancer/tile_morph_clustering/tile_clustering"

python cluster_tiles.py \
    --pt-dir "$PT_DIR" \
    --tiles "$TILES" \
    --morphology "$MORPH" \
    --ks 2,4,8,16,32 \
    --max-tiles 200000 \
    --out "$OUT"
