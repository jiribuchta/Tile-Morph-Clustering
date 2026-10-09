import csv
import sys

KEEP_PREFIXES = ("850", "852")

def main():
    src = sys.argv[1] if len(sys.argv) > 1 else "slides.csv"
    out = sys.argv[2] if len(sys.argv) > 2 else "output.csv"

    with open(src, newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = [row for row in reader if row["morphology"].startswith(KEEP_PREFIXES)]

    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"kept {len(rows)} rows -> {out}")

if __name__ == "__main__":
    main()
