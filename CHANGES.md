# CHANGES — cleanup-submission-v2

All tasks completed on branch `cleanup-submission-v2`. No tasks fell back to fallbacks.

---

## Task 1 — Repo housekeeping
- `git mv data/labels/gold_dataset_binary.csv data/labels/archive/gold_dataset_binary.csv`
- Created `data/labels/archive/README.md` clarifying that `gold_dataset_binary.csv` is a superseded pilot artifact and that `test_set_final.csv` (28 HIGH / 32 LOW) is the canonical evaluation set.
- Grepped root `README.md` for `gold_dataset_binary` and "20% HIGH" — no stale references found; README left unchanged.

## Task 2 — PDF font fix
Added to preamble immediately after `\usepackage[T1]{fontenc}`:
```latex
\usepackage{lmodern}
\usepackage{microtype}
```

## Task 3 — Removed manifesto / overclaim language
- **3a**: Deleted the "The problem extends beyond individual learning loss…" paragraph in the Introduction (info-warfare/election-integrity overclaim).
- **3b**: Deleted the `\textbf{Information integrity and manipulation detection.}` subsection in Future Work.
- **3c**: Deleted the sentence "The annotation process is also legally clean: labels are assigned to transcript content by rubric criteria, not to creators by name or reputation."
- **3d**: Changed "platform tools eliminate it" → "platform tools reduce it" in the two-mode deployment paragraph.

## Task 4 — Fixed HIGH-signal definition contradiction
Changed:
> "A transcript is classified as HIGH signal if it satisfies **all** of the following criteria:"

to:
> "A transcript is classified as HIGH signal if it satisfies **at least two** of the following criteria:"

This matches the per-domain rubric definitions in Appendix A.

## Task 5 — Transcript-count consistency
- Abstract updated: 150 collected, 149 received usable labels (1 excluded), arithmetic 89 + 60 = 149 stated explicitly.
- Dataset Construction section updated: "Training supplement (90 collected; 89 retained after one exclusion)" with the 89+60=149 accounting.
- Limitations section: "90 real training transcripts" → "89 real training transcripts".

## Task 6 — Citation fixes
- **6a**: Fixed `pitler2008` title from "…predicting age of acquisition and lexical decision times" → "…predicting text quality." (EMNLP 2008, pages 186–195, authors unchanged.)
- **6b**: Replaced `\cite{lipton2018}` (interpretability paper, wrong context) with `\cite{hegarcia2009}` (He & Garcia 2009, IEEE TKDE — correct reference for imbalanced classification evaluation). Removed the `lipton2018` bibitem (no other uses found). Added `hegarcia2009` bibitem.

## Task 7 — Synthetic-only baseline table (RUN COMPLETED)
Script: `experiments/run_ablations.py`
Results saved to: `reports/classifier_results_synthetic_only_finaltest.json`

Numbers produced (evaluated on `test_set_final.csv`, n=60):

| Model | Acc | F1 | Prec | Rec |
|---|---|---|---|---|
| SVM RBF (synthetic-only, thresh=0.50) | 0.683 | 0.747 | 0.596 | 1.000 |
| LR (synthetic-only, thresh=0.50) | 0.683 | 0.732 | 0.605 | 0.929 |

Paper change: Table 1 now includes these synthetic-only rows (under a "Synthetic-only training" subheader) and the results narrative cites the F1 improvement from 0.747 → 0.871. No fallback taken.

## Task 8 — No-domain-feature ablation (RUN COMPLETED)
Results saved to: `reports/ablation_no_domain.json`

Numbers produced (SVM-RBF, 384-dim embeddings only, full mixed train, test_set_final.csv):

| Model | Acc | F1 | Prec | Rec |
|---|---|---|---|---|
| SVM RBF (no domain, thresh=0.50) | 0.833 | 0.844 | 0.750 | 0.964 |

Paper change: Added `\subsection{Domain Feature Ablation}` to Results reporting these numbers and noting the modest F1 drop (0.871 → 0.844) suggests the classifier is not primarily exploiting domain priors. No fallback taken.

## Task 9 — Reframed F1=0.871 as feasibility / silver-label agreement
- Abstract: added "(measured against silver labels; this reflects internal consistency with the labeling pipeline rather than human-validated accuracy)" on first statement of F1=0.871. Also replaced "substantially improve … relative to synthetic-only training baselines" with the concrete number "improve … (synthetic-only SVM achieves F1=0.747 on the same test set)".
- Conclusion: same silver-label qualifier added inline; "substantially improves performance" softened to "improves performance"; synthetic-only F1=0.747 baseline cited.

## Task 10 — Softened integrity-timing claim
Replaced the Dataset Integrity Protocol paragraph to remove the unverifiable "labels committed prior to training" ordering claim. New text acknowledges version-control as the audit trail and states "Labels were not modified after publication."

---

## Build verification
```
pdflatex snr_detector_v2.tex  (pass 1) → 12 pages, no errors
pdflatex snr_detector_v2.tex  (pass 2) → 12 pages, no errors
```
Output: `papers/snr_detector_v2.pdf`
