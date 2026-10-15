"""Match a list of slide names against the morphology CSV on record (slide_name.split('-')[0]).

Usage:
    python match_morphology.py slides.txt morphology.csv [--out slide_morphology.csv]

slides.txt: one slide name per line (e.g. 2020_00172-04-T)
morphology.csv: record,side,morphology
"""
import argparse
import csv
import re
from collections import Counter, defaultdict

SLIDE_RE = re.compile(r"^\d{4}_\d{5}-\d{1,3}-[NT]\d?$")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("slides", help="file with one slide name per line")
    ap.add_argument("morph_csv", help="morphology CSV (record,side,morphology)")
    ap.add_argument("--out", default="slide_morphology.csv")
    args = ap.parse_args()

    # morphology: record -> {side: morphology}
    morph = defaultdict(dict)
    with open(args.morph_csv, newline="") as f:
        for row in csv.DictReader(f):
            morph[row["record"].strip()][row["side"].strip()] = row["morphology"].strip()

    rows, unmatched, malformed = [], [], []
    for ln in open(args.slides):
        name = ln.strip()
        if not name:
            continue
        if not SLIDE_RE.match(name):
            malformed.append(name)
            continue
        rec = name.split("-")[0]
        sides = morph.get(rec)
        if sides is None:
            unmatched.append(name)
            continue
        vals = set(sides.values())
        m = next(iter(vals)) if len(vals) == 1 else " | ".join(f"{k}:{v}" for k, v in sorted(sides.items()))
        rows.append((name, rec, m))

    rows.sort()
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["slide_name", "record", "morphology"])
        w.writerows(rows)

    c = Counter(r[2] for r in rows)
    print(f"{len(rows)} slides matched -> {args.out}")
    for m, n in sorted(c.items()):
        print(f"  {n:>4}  {m}")
    if unmatched:
        print(f"{len(unmatched)} slides with no morphology record:")
        for name in unmatched:
            print(f"  {name}")
    if malformed:
        print(f"{len(malformed)} malformed lines (skipped):")
        for name in malformed:
            print(f"  {name}")


if __name__ == "__main__":
    main()
