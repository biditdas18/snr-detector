# CHANGES — cleanup-submission-v5

All tasks completed. No fallbacks.

## Task 1 — Abstract: "significant" → "substantial"
Changed "82% average pairwise agreement---a significant improvement over the 53%" to
"---a substantial improvement over the 53%". Only this occurrence changed.

## Task 2 — §4.1 LLM Ensemble Annotation: remove rubric-only causality
Replaced "The calibrated domain-specific rubrics improved average pairwise inter-model
agreement from 53% (domain-agnostic) to 82% across domains, with 74% of transcripts achieving
full three-model consensus." with "The production labeling setup—combining calibrated
domain-specific rubrics with the final judge ensemble—achieved 82% average pairwise agreement
(74% full three-model consensus), compared with 53% in the initial domain-agnostic setup."

## Task 3 — Appendix B: fix garbled duplication
Removed the duplicated "Labels are / Labels follow" fragment; sentence now reads cleanly:
"Labels follow the generation prompt: HIGH prompts are intended to produce HIGH-signal
transcripts, though prompt adherence is not guaranteed (see Section 4.2)."

## Task 4 — Yurdakul year: already correct (14(4), 2020). No change.

## Task 5 — zheng2023 page numbers added (optional)
Added "pages 46595--46623" to MT-Bench NeurIPS entry. Build clean.

## Build verification
pdflatex x2 → 12 pages, no errors, all 12 citations resolved.

---

# CHANGES — cleanup-submission-v4

All tasks completed on branch `cleanup-submission-v4`. No tasks fell back.

---

## Task 1 — Cite companion paper (§6.3 + bibitem)
Added `\cite{crucible}` inline: "local calibration panel reported in a companion paper~\cite{crucible}".
Added `crucible` bibitem after the Statista entry (repo-only citation; no DOI/arXiv invented).

## Task 2 — "immutable" → "timestamped" (§4.3)
Changed "providing an immutable, publicly auditable record." to
"providing a timestamped, publicly auditable record."

## Task 3 — Appendix B "by definition" softened
Changed "Labels are assigned by construction from the generation prompt---HIGH prompts produce
HIGH transcripts by definition." to "Labels follow the generation prompt: HIGH prompts are
intended to produce HIGH-signal transcripts, though prompt adherence is not guaranteed
(see Section 4.2)."

## Task 4 — Yurdakul volume/year corrected
Changed "Journal of Risk Model Validation, 2019." to
"Journal of Risk Model Validation, 14(4), 2020." (journal-of-record issue, not online-first).

## Task 5 — Abstract causality softened
Changed "We demonstrate that domain-specific rubric calibration and real-data augmentation
improve…" to "We show that the combination of domain-specific rubrics, a production labeling
ensemble, and real-data augmentation improves…"

## Task 6 — Conclusion causality softened
Changed "a domain-specific rubric calibration procedure that substantially improves LLM ensemble
annotation reliability" to "a domain-specific rubric calibration procedure that, together with
the production labeling ensemble, contributes to improved LLM ensemble annotation reliability."

---

## Build verification
```
pdflatex snr_detector_v2.tex  (pass 1) → 12 pages, no errors
pdflatex snr_detector_v2.tex  (pass 2) → 12 pages, no errors, all \cite resolved
```
Total references: 12 (added crucible as [12]). Output: papers/snr_detector_v2.pdf

---

# CHANGES — cleanup-submission-v3

All tasks completed on branch `cleanup-submission-v3`. No tasks fell back.

---

## Task 0 — Rendering setup verified
`\usepackage{lmodern}` and `\usepackage{microtype}` confirmed present in preamble (added in v2
pass). Build is clean; no further change needed for the missing-"i" rendering complaint.

## Task 1 — Softened rubric-calibration causality (3 locations)

**1a (Abstract):** Changed "…over the 53% pairwise agreement obtained with an initial
domain-agnostic rubric." to "…obtained in the initial domain-agnostic setup (which differed in
both rubric and judge ensemble)."

**1b (Section 6.3 — Annotation Reliability):** Replaced the single "directly validates" sentence
with a full paragraph acknowledging the confounded effect (rubric + judge ensemble both changed;
82% is the production frontier ensemble; Career domain did not converge in the local calibration
panel).

**1c (Conclusion):** Replaced "We demonstrated that domain-specific rubric calibration resolves
near-random LLM annotation agreement (53% → 82% pairwise)" with "We showed that moving to
domain-specific rubrics together with the production labeling ensemble raised pairwise
inter-model agreement from 53% to 82%."

## Task 2 — Added confidence intervals (Limitations, "Small real test set")
Replaced "Confidence intervals are not reported due to test set size." with 95% Wilson score
intervals from the 25/7/1/27 confusion matrix:
accuracy 0.867 [0.758, 0.931], recall 0.964 [0.823, 0.994], precision 0.794 [0.632, 0.897].

**Bootstrap F1 CI (optional):** SKIPPED — no per-item test-prediction file found in `reports/`.
Sentence retains "a bootstrap interval for F1 … scoped to future work."

## Task 3 — Softened synthetic "perfect label reliability" (Section 4.2)
Replaced "This eliminates annotation ambiguity and ensures perfect label reliability…" with
language noting prompt-adherence errors remain possible; synthetic labels are "high-confidence
but not error-free."

## Task 4 — Fixed yurdakul2018 citation
Replaced incorrect arXiv:1809.04233 / single-author entry with:
Bilal Yurdakul and Joshua D. Naranjo. Statistical properties of the population stability index.
*Journal of Risk Model Validation*, 2019.

## Task 5 — Added citation for 500-hours claim
Added `\cite{statista_youtube}` in Introduction after "over 500 hours of video are uploaded
every minute." Added `statista_youtube` bibitem (Statista, URL, accessed June 2026).

## Task 6 — Softened LLM-bias overclaim (Section 4)
Changed the "Language models apply the rubric without creator recognition or domain preference"
sentence to acknowledge LLM annotation *reduces* (not eliminates) human annotator bias, and
that model-level pretraining bias remains — consistent with silver-label framing.

---

## Build verification
```
pdflatex snr_detector_v2.tex  (pass 1) → 12 pages, no errors
pdflatex snr_detector_v2.tex  (pass 2) → 12 pages, no errors, all \cite resolved
```
Total references: 11 (added statista_youtube). Output: papers/snr_detector_v2.pdf

---

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
