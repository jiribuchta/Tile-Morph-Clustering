"""Diagnose the xOpat mask<->WSI mismatch (run ON THE CLUSTER).

Usage:  python scripts/diagnose_report.py [path/to/generated report_conf/.../tile_morph_k32.yaml]

Finds the generated report conf, then compares the WSI filenames xOpat shows
against the mask filenames it overlays, and prints one slide's WSI vs mask
level-0 dims + MPP. If the stems don't match 1:1, that IS the mismatch.
"""
import re
import sys
from pathlib import Path


def main() -> None:
    rc = sys.argv[1] if len(sys.argv) > 1 else None
    if rc is None:
        # prefer a GENERATED conf (has an explicit `paths:` block) over the template
        cands = [str(p) for p in Path().glob("**/report_conf/reporter/tile_morph_k32.yaml")]
        gen = [c for c in cands if "paths:" in Path(c).read_text(errors="ignore")]
        if not gen:
            print("pass the generated report conf path as arg 1. candidates:\n  " + "\n  ".join(cands or ["(none)"]))
            return
        rc = gen[0]
    text = Path(rc).read_text(errors="ignore")
    print("report conf :", rc)

    masks_dir = re.search(r"dir_name:\s*(\S+)", text)
    masks_dir = Path(masks_dir.group(1)) if masks_dir else Path(".")
    print("masks dir   :", masks_dir)

    # WSI list: SlideRetriever `paths:` -> indented `- <path>` items
    wsis = [w.strip() for w in re.findall(r"^\s+-\s*(\S+)", text, flags=re.M)
            if Path(w).suffix.lower() in {".mrxs", ".svs", ".ndpi", ".tif", ".tiff"}]
    masks = sorted(p.stem for p in masks_dir.glob("*.tiff"))
    wsi_stems = [Path(w).stem for w in wsis]
    on_disk = [w for w in wsis if Path(w).exists()]

    print(f"\nn WSI paths in conf : {len(wsis)}")
    print(f"n WSIs on disk      : {len(on_disk)}")
    print(f"n masks on disk     : {len(masks)}")

    only_mask = sorted(set(masks) - set(wsi_stems))
    only_wsi = sorted(set(wsi_stems) - set(masks))
    both = sorted(set(masks) & set(wsi_stems))
    print("\nSTEM MATCH  (mask<->WSI join key is the file stem):")
    print(f"  matched        : {len(both)}")
    print(f"  mask-only      : {len(only_mask)}  {only_mask[:8]}")
    print(f"  wsi-only       : {len(only_wsi)}  {only_wsi[:8]}")
    if only_mask or only_wsi:
        print("  ^^ MISMATCH: those stems won't line up in xOpat.")
    if both:
        print(f"  e.g. matched stems: {both[:8]}")

    # one concrete slide: WSI vs mask level-0 dims + MPP
    for stem in both[:1]:
        wsi = next((w for w in wsis if Path(w).stem == stem), None)
        mask = masks_dir / f"{stem}.tiff"
        print(f"\nSLIDE {stem!r} dims + MPP:")
        import openslide
        for label, p in [("WSI ", wsi), ("MASK", str(mask))]:
            try:
                s = openslide.OpenSlide(p)
                print(f"  {label} L0={tuple(s.level_dimensions[0])} "
                      f"mpp=({s.properties['openslide.mpp-x']}, {s.properties['openslide.mpp-y']})")
                s.close()
            except Exception as e:
                print(f"  {label} FAILED to open {p}: {e}")


if __name__ == "__main__":
    main()
