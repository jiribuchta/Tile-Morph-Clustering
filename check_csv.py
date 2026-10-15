"""Report which slides are in a (possibly partial) heatmap CSV.

Usage: python check_csv.py heatmaps_tiles.csv
"""
import csv
import sys
from collections import Counter

c = Counter()
with open(sys.argv[1], newline="") as f:
    r = csv.reader(f)
    next(r)  # header
    for row in r:
        c[row[1]] += 1  # slide_name

total = sum(c.values())
print(f"{len(c)} slides, {total} rows")
for name, n in sorted(c.items()):
    print(f"  {n:>10}  {name}")
