# SNR-Detector: AI Agent Coding Guide

## Architecture Overview

**SNR-Detector** is a deterministic Signal-to-Noise Ratio scoring system for social media content that extracts interpretable text features and applies lightweight calibration. The system has three core phases:

1. **Feature Extraction** → Anchor features (physics-based text metrics)
2. **Dual Scoring** → Baseline ML (Ridge) + SNR Engine (clarity/depth/penalty components)
3. **Calibration** → 3-weight Ridge layer to harmonize baseline and engine

**Data Flow:** Raw transcript → Anchor features (deterministic, no labels needed) → Baseline/Engine scores (deterministic formulas) → Calibration (Ridge, trained on small gold seed set)

## Key Components & Patterns

### 1. Anchor Features (Deterministic Text Metrics)
**File:** [scripts/build_anchor_features.py](scripts/build_anchor_features.py)

These are intentionally low-level, reproducible text measurements (no NLP models, no ML training):
- `word_count`, `entropy`, `unique_ratio` — text volume/diversity
- `recycled_similarity`, `recycled_penalty` — TF-IDF cosine distance to `data/generic_advice_bank.txt`
- `fear_hits`, `promo_hits`, `hype_hits`, `evidence_hits` — regex pattern matches from dictionary lists
- `structure_hits`, `qmark_count`, `exclam_count` — punctuation and sentence structure
- `modal_count`, `you_count`, `imperative_count` — linguistic patterns

**Critical constraint:** All features must be strictly deterministic and indexable (no models). The `build_recycled_similarity_fn()` uses TF-IDF fitted once per run to avoid drift.

### 2. Baseline Model
**File:** [scripts/train_snr_baseline.py](scripts/train_snr_baseline.py)

Ridge regression trained on the full feature set (see `ANCHOR_FEATURES` allowlist). Requires exact feature list—missing columns raise `ValueError`. Cross-validates with configurable `--alpha` and `--folds`.

**Pattern:** Always validate required columns before training; use median imputation + standard scaling.

### 3. SNR Engine (Decomposed Scoring)
**File:** [scripts/compute_snr_engine.py](scripts/compute_snr_engine.py)

Manually weighted sub-components (no training):
- **Clarity:** `0.4*structure_hits + 0.4*evidence_hits + 0.2*content_density_proxy`
- **Depth:** `0.5*entropy_norm + 0.3*unique_ratio + 0.2*avg_sent_len`
- **Penalty:** `0.5*fear_hits + 0.3*promo_hits + 0.2*recycled_penalty`

Z-scores are computed against the **input batch** or a **reference CSV** (use `--ref_csv` to normalize single-video scoring).

**Pattern:** All components are z-scored to a common scale [~-2, +2]. Penalties are inverted (higher = worse). Design principle: interpretability over perfect predictive power.

### 4. Calibration Layer
**File:** [scripts/calibrate_engine.py](scripts/calibrate_engine.py)

Trains a simple Ridge on `clarity_z, depth_z, penalty_z` to predict `snr_score`. Output weights show relative importance (e.g., clarity has positive weight, penalty has negative).

**Pattern:** Always fit and save coefficients; report cross-validation MAE/RMSE/R². Used in `score_youtube.py` for user-facing predictions.

## Workflows & Command Patterns

### Full Pipeline (Known-Good Gold Set)
```bash
# 1. Build anchor features from labeled gold data
python scripts/build_anchor_features.py \
  --input data/labels/gold_labels_llm_snrC.csv \
  --output reports/anchor_features_v2.csv \
  --generic-advice-bank data/generic_advice_bank.txt

# 2. Train baseline (Ridge)
python scripts/train_snr_baseline.py \
  --input reports/anchor_features_v2.csv \
  --alpha 100 --folds 3 \
  --out_coef reports/baseline_v2/coef_a100_f3.csv \
  --out_pred reports/baseline_v2/pred_a100_f3.csv

# 3. Compute engine scores (no training)
python scripts/compute_snr_engine.py \
  --input reports/anchor_features_v2.csv \
  --output reports/snr_engine_scores.csv

# 4. Calibrate engine (fit Ridge on z-scores)
python scripts/calibrate_engine.py \
  --engine_csv reports/snr_engine_scores.csv \
  --alpha 10 --folds 3 \
  --out_pred reports/calibrated_engine_preds_a10_f3.csv \
  --out_coef reports/calibrated_engine_weights_a10_f3.csv
```

### Single-Video Scoring (Any YouTube Video ≤ 30 min)
**File:** [scripts/score_youtube.py](scripts/score_youtube.py)

Downloads transcript, extracts features, computes scores. Full workflow in README (Step A–D). Key constraints:
- Requires `yt-dlp` (install: `brew install yt-dlp`)
- Uses reference z-score normalization (`--ref_csv`) so single videos compare meaningfully
- Output includes interpretable labels ("HIGH", "MID", "LOW") based on `classify_from_engine()`

## Project-Specific Conventions

### Feature Allowlisting
**Pattern:** Both baseline and engine explicitly list required features, fail hard if missing.
- Baseline: `ANCHOR_FEATURES` in [scripts/train_snr_baseline.py](scripts/train_snr_baseline.py#L8)
- Engine: computed from `clarity`, `depth`, `penalty` components

**Why:** Ensures reproducibility; catch renamed or missing columns early.

### Generic Advice Bank
**File:** [data/generic_advice_bank.txt](data/generic_advice_bank.txt)

One phrase per line. Used to compute `recycled_similarity` (TF-IDF cosine to these patterns). If empty or malformed, anchor feature extraction fails immediately.

**Pattern:** Immutable reference list. Changes to it require re-running feature extraction on all data.

### Gold Seed Labels
**Location:** [data/labels/](data/labels/)

Multiple CSV variants (`gold_labels_llm_snrC.csv`, `gold_labels_llm_snrB.csv`, etc.) all share the same schema:
- Required: `video_id`, `title`, `transcript` (text column name varies), `snr_score` (label)
- Optional: `takeaway_clarity_1_5`, `insight_depth_1_5`, etc. (metadata, ignored by most scripts)

**Pattern:** Always validate input CSV schema before processing; raise informative errors on missing required columns.

### CSV Output Schema
Anchor features output includes:
- `video_id`, `title`, `snr_score` (original columns)
- All `ANCHOR_FEATURES` (e.g., `word_count`, `entropy`, `recycled_similarity`, etc.)

Engine output adds:
- `clarity_z`, `depth_z`, `penalty_z` (z-scored components)
- `snr_engine_score` (combined score, higher = more signal)

Calibrated output adds:
- `snr_engine_calibrated_pred_trainfit` (final prediction from Ridge)

**Why:** Downstream consumers expect exact column names. Use `[["col1", "col2", ...]].copy()` when subsetting.

## Integration & Extension Points

### Adding a New Feature
1. Implement deterministic computation in [scripts/build_anchor_features.py](scripts/build_anchor_features.py)
2. Add to `ANCHOR_FEATURES` list in [scripts/train_snr_baseline.py](scripts/train_snr_baseline.py)
3. If it improves engine clarity/depth/penalty, update the formula in [scripts/compute_snr_engine.py](scripts/compute_snr_engine.py)
4. Re-run full pipeline to update reports/

### Modifying Engine Weights
Edit the hardcoded weights in `compute_clarity()`, `compute_depth()`, `compute_penalty()` ([scripts/compute_snr_engine.py](scripts/compute_snr_engine.py#L24)).
Then re-run steps 3–4 of the pipeline above.

**Caveat:** Weights are intentionally hand-tuned for interpretability, not optimized. Changing them may require recalibrating.

### Model Tuning
- **Baseline alpha:** Test via `--alpha` flag; higher values regularize more (default 1.0). Try [0.1, 1, 10, 100] for ablation.
- **Calibration alpha:** Similar; controls how much the engine z-scores are trusted vs. baseline predictions.
- **Cross-validation folds:** Use `--folds` to set (default 5); fewer folds with small datasets.

## Dependencies & External Tools

- **Core ML:** `scikit-learn` (Pipeline, Ridge, StandardScaler, ColumnTransformer, KFold)
- **Data:** `pandas`, `numpy`
- **Text:** `nltk` (tokenization), `sklearn.feature_extraction.text` (TF-IDF)
- **Transcripts:** `youtube-transcript-api`, `yt-dlp` (for YouTube video downloads)
- **Serialization:** `joblib` (model pickling in [src/snr_classifier.py](src/snr_classifier.py))

## Testing & Validation

- **Determinism:** All feature extraction is fully deterministic (same input → same features). Verify by running `build_anchor_features.py` twice on the same CSV and diff outputs.
- **Data validation:** All scripts raise `ValueError` on missing required columns. Catch these early in test scripts.
- **CV metrics:** Baseline and calibration report MAE, RMSE, R² on each fold. Use these to catch overfitting (R² ≈ 1.0 on small gold sets is normal but suspect).

## File Organization

```
src/
  snr_classifier.py      — SNRModel dataclass, Ridge pipeline, serialization
  preprocess.py          — Helper to invoke build_anchor_features.py on temp CSVs
  transcript_fetcher.py  — YouTube URL parsing, transcript API calls
data/
  generic_advice_bank.txt — Immutable reference phrases for recycled-signal penalty
  labels/                — Gold labeled CSVs (various versions)
  transcripts/           — Transcript scrapes (CSV format)
scripts/
  build_anchor_features.py — Feature extraction (deterministic, no labels)
  train_snr_baseline.py  — Ridge training on full feature set
  compute_snr_engine.py  — Engine scoring (clarity/depth/penalty, no training)
  calibrate_engine.py    — Calibration Ridge (predicts snr_score from z-scores)
  score_youtube.py       — End-to-end: download + featurize + score any video
reports/
  — Artifacts: CSVs with features, coefficients, predictions
```

---

## When Debugging

1. **Feature mismatch:** Check anchor features CSV schema. Missing columns → check `ANCHOR_FEATURES` allowlist.
2. **Z-score issues:** If single-video z-scores seem extreme, provide `--ref_csv` to normalize against a known distribution.
3. **Calibration not improving baseline:** Engine might be capturing orthogonal information. Check correlation between baseline & engine scores in reports/.
4. **Transcript download fails:** Verify yt-dlp is installed and video has downloadable captions (not all videos do).
