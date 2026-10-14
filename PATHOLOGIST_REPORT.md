# Findings: AI-based ductal vs lobular morphology from WSI tile embeddings

**Prepared for pathologist review — v5, October 2025**

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

| Slide-level metric (Val, n=293) | value |
|---|---|
| AUC (ranking: lobular scores above ductal) | **0.878** |
| Accuracy at the natural threshold (score > 0) | 81.2% |
| Balanced accuracy | ~83% |

Interpretation: given one random ductal and one random lobular slide, the model
ranks them correctly ~88% of the time. This is from tile morphology alone.

## The interesting part: the structure of the errors

55/293 slides land on the wrong side of the threshold. When we looked at
them, they split into three distinct groups:

1. **Half of Val has almost no detectable tumor tiles.** The carcinoma
detector (probability >= 0.95) finds thousands of tumor tiles on most
slides, but on ~25% of slides it finds fewer than ~16 tiles — some just one.
A slide score averaged over a handful of tiles is noise, and 7 of our 9
strongest "contradictions" were exactly such slides. These are cases where
the tumor was likely missed by the detector (or sampled by a different
slide), **not** morphology errors. With such slides excluded (fewer than 100
tumor tiles), the headline numbers improve.

2. **Two well-sampled slides contradict the label.**
`2020_08267-04-T2` (8520 labeled, 27k tumor tiles, mean score -1.48 = solidly
ductal) and `2023_08660-2-T` (8500 labeled, 46k tiles, mean +1.63 = solidly
lobular). These are the genuine contradictions: mixed tumors or morphology
coding differences in the clinical table.

3. **The remaining ~45 errors sit near the boundary (|score| < 0.5)** —
genuinely ambiguous slides, exactly what overlapping distributions produce.

Our hypothesis for group 2: the label records only the dominant component;
a mixed ductal-lobular tumor scored tile-by-tile can read as the minority
morphology. If so, the true separation is better than the reported AUC —
we are grading against imperfect ground truth.

## What we ask you to look at

An xOpat report with the model's per-tile score overlaid on each WSI:
**blue = ductal-leaning tiles, red = lobular-leaning tiles**, uncertain middle
band hidden (can be re-enabled in the viewer). Empty-looking slides are those
where the detector found (almost) no tumor tiles. Please review the two
genuine contradictions first, then a sample of ordinary slides:

| slide | labeled | model score | tumor tiles | notes |
|---|---|---|---|---|
| 2020_08267-04-T2 | 8520 | −1.48 | 27,455 | solidly ductal despite lobular label; sibling slide 2020_08267-03-N2 has almost no detectable tumor |
| 2023_08660-2-T | 8500 | +1.63 | 45,962 | solidly lobular despite ductal label |

Three questions per slide:

1. **Calibration** — do the blue regions correspond to ductal-pattern
   epithelium and the red regions to lobular-pattern epithelium (single-file
   growth, loss of E-cadherin-type discohesion)?
2. **The contradictions** — for the two slides above: is the slide genuinely
   the labeled morphology (model error), a mixed tumor, or mislabeled?
3. **The empty slides** — pick a few slides where the mask shows almost no
   colored tiles: is there actually tumor on them? (These test the carcinoma
   detector, not the morphology axis.)

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
