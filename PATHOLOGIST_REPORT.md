# Findings: AI-based ductal vs lobular morphology from WSI tile embeddings

**Prepared for pathologist review — v6, October 2025**

*v6 note: all scores were regenerated after fixing a tile-alignment bug in
the scoring pipeline (tile scores were written at the wrong coordinates
when filtering to carcinoma tiles). The pathologist review below was
performed on the v5 masks; both of its verdicts were re-verified and hold
under v6 — with stronger scores.*

## The question

Can a computer vision model reading only H&E tile morphology — no clinical data,
no spatial context, no gross description — tell invasive ductal carcinoma
(ICD-O 8500) from invasive lobular carcinoma (8520), tile by tile?

## Method in one paragraph

Each whole-slide image is divided into ~224 px tiles. Every tile gets a
Virchow2 embedding (a pretrained pathology foundation model). A separate
carcinoma detector first selects tumor tiles (probability ≥ 0.95 — carcinoma
epithelium only, stroma and normal tissue excluded). On the selected tiles we
trained a linear projection (LDA) on 1,435 training slides (1,189 ductal,
246 lobular) to find the direction in embedding space that best separates the
two morphologies. Every tile then gets a single score:

- **negative = ductal-like, positive = lobular-like, 0 = decision boundary**
- slide score = mean over its carcinoma tiles
- the scale is global: a score of +1.5 means the same thing on every slide

The direction was trained on a separate slide set (Train); all numbers below
are from **Val — 293 labeled slides the model never saw** (248 ductal, 45
lobular).

## Headline result

Slides with sufficient detected tumor (>= 100 carcinoma tiles; 196 of 293
labeled Val slides — the excluded 97 have a median of **4** detected tumor
tiles, i.e. essentially no tumor found on them):

| Slide-level metric (Val, n=196) | value |
|---|---|
| AUC (ranking: lobular scores above ductal) | **0.976** |
| Accuracy at the natural threshold (score > 0) | **98.0%** |
| Balanced accuracy | 96.2% |

Interpretation: given one random ductal and one random lobular slide, the model
ranks them correctly ~98% of the time. This is from tile morphology alone.

(Caveat: the lobular side is 31 slides, so the AUC carries a confidence
interval of roughly ±0.03.)

## The interesting part: the structure of the errors

97 of the 293 Val slides carry almost no detectable tumor (median 4
carcinoma tiles) — mostly `N` (negative) blocks per the bioptic-tree
naming scheme (`T` = tumor, `N` = negative, `M` = metastases) that inherit
their case's 8500/8520 label via the record-prefix join. Their mean score
is noise, so they are excluded (< 100 tiles); the headline numbers above are
the remaining 196 slides.

Of those 196, only **4** land on the wrong side of the threshold:

1. **Two are the known label errors** (see the review outcome below):
   `2020_08267-04-T2` (8520 labeled, 27k tumor tiles, mean score −3.28 =
   solidly ductal) and `2023_08660-2-T` (8500 labeled, 46k tiles, mean
   +3.90 = solidly lobular). Pathologist review confirmed both are table/
   coding errors; the model was right on both.
2. **Two sit on the boundary**: `2021_05355-02-T` (8520 labeled, mean
   −0.20) and `2024_00961-2-N` (8500 labeled, mean +0.05 — an `N` block
   inheriting its case label, 580 tiles). Genuinely ambiguous, exactly
   what overlapping distributions produce.

In other words: on well-sampled slides the model has **no unexplained
errors** — every wrong call is a documented label error or a coin-flip
at the decision boundary. We are grading against imperfect ground truth.

## Outcome of the pathologist review

Both genuine contradictions were **label errors, not model errors**:

- `2020_08267-04-T2` — patient has multiple tumors in both breasts; this
  block is from the right-breast carcinoma and is genuinely ductal (8500).
  The table had it assigned to the (lobular) left-breast tumor; the sibling
  T1 block is the 8520 one. Model was right (v6 score −3.28).
- `2023_08660-2-T` — genuinely lobular carcinoma; a coding error whose
  correction never propagated into the database export. Model was right
  (v6 score +3.90).

With both resolved, the model has no unexplained contradictions on
well-sampled slides — every remaining error sits near the decision
boundary.

**Open question:** on mixed-labeled tumors, the pathologist could not
visually define a difference between the blue- and red-colored tiles. Note
that this review was done on the v5 masks, which carried the tile-alignment
bug — the blue/red values sat at wrong coordinates, which may itself
explain why no visual pattern was found. The v6 masks put each score at
its true tile; whether the extreme-scoring tiles are visually distinct
deserves a fresh look. The axis clearly separates slide-level labels
(AUC 0.976), but its per-tile evidence lives in 2560-dimensional
foundation-model features, which need not correspond to the architectural
patterns humans name — the signal may be subtle texture/statistics rather
than single-file vs gland-forming growth.

## What we ask you to look at

An xOpat report with the model's per-tile score overlaid on each WSI:
**blue = ductal-leaning tiles, red = lobular-leaning tiles**, uncertain middle
band hidden (can be re-enabled in the viewer). Empty-looking slides are those
where the detector found (almost) no tumor tiles. Please review the two
genuine contradictions first, then a sample of ordinary slides:

| slide | labeled | model score | tumor tiles | notes |
|---|---|---|---|---|
| 2020_08267-04-T2 | 8520 | −3.28 | 27,455 | solidly ductal despite lobular label; sibling slide 2020_08267-03-N2 has almost no detectable tumor |
| 2023_08660-2-T | 8500 | +3.90 | 45,962 | solidly lobular despite ductal label |

Three questions per slide:

1. **Calibration** — do the blue regions correspond to ductal-pattern
   epithelium and the red regions to lobular-pattern epithelium (single-file
   growth, loss of E-cadherin-type discohesion)?
2. **The contradictions** — for the two slides above: is the slide genuinely
   the labeled morphology (model error), a mixed tumor, or mislabeled?
3. **The empty slides** — the naming scheme says these are mostly `N`
(negative) blocks; if any mask on an `N` slide shows substantial colored
tumor regions, that is a carcinoma-detector false positive worth checking.
`M` (metastases) slides would be the opposite check.

## Bonus observation

Slides with rarer morphologies (not used in training) score coherently on the
same axis: e.g. lobular + DCIS (8522) and papillary (8509) lean lobular;
tubular (8503), cribriform (8504) lean ductal. The axis seems to capture
growth-pattern biology, not just the two training classes.

## Technical details available on request

Virchow2 embeddings; carcinoma tile filter ≥ 0.95; LDA with shrinkage and
balanced priors; training set Train (B20–24), test set Val; per-slide tile cap
500 for training. Full eval output, score CSVs, and masks are under
`/mnt/projects/breast_cancer/tile_morph_clustering/`.
