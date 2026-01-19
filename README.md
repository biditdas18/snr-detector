# SNR-Detector  
**Signal-to-Noise Ratio Detection for Social Media Content**

> A deterministic, interpretable system to distinguish **informative signal** from **high-confidence noise** in social-media style content, using transcript-derived anchor features and a lightweight calibration layer.

---

## Why this exists

Modern social platforms reward:
- urgency over understanding  
- repetition over novelty  
- fear/promo over clarity  

This project scores content by **what it adds** (signal) vs **what it inflates** (noise), with a strong bias toward:
- reproducibility
- interpretability
- research-grade artifacts

---

## What you get in this repo

- **Anchor (physics) features** extracted deterministically from transcript text.
- **Baseline ML** (Ridge regression) trained on a small gold seed set (documented limitations).
- **Decomposed SNR Engine** (clarity/depth/penalty) + **calibration** (3 weights + bias).
- **Artifacts** under `reports/` that let you compare approaches side-by-side.

---

## Non-technical quickstart (3–4 steps)

This is the fastest way for a recruiter (or any non-ML person) to run the pipeline and verify it works.

### Constraints
- YouTube video must have a downloadable transcript/captions.
- Keep the video **≤ 30 minutes** (long transcripts can make results less reliable for this prototype).

### Step 1 — Setup
```bash
git clone <YOUR_REPO_URL>
cd snr-detector

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Step 2 — Run the full pipeline on the included gold seed set (known-good)
```bash
mkdir -p reports/baseline_v2

python scripts/build_anchor_features.py \
  --input data/labels/gold_labels_llm_snrC.csv \
  --output reports/anchor_features_v2.csv \
  --generic-advice-bank data/generic_advice_bank.txt

python scripts/train_snr_baseline.py \
  --input reports/anchor_features_v2.csv \
  --alpha 100 \
  --folds 3 \
  --out_coef reports/baseline_v2/coef_a100_f3.csv \
  --out_pred reports/baseline_v2/pred_a100_f3.csv

python scripts/compute_snr_engine.py \
  --input reports/anchor_features_v2.csv \
  --output reports/snr_engine_scores.csv

python scripts/calibrate_engine.py \
  --engine_csv reports/snr_engine_scores.csv \
  --alpha 10 \
  --folds 3 \
  --out_pred reports/calibrated_engine_preds_a10_f3.csv \
  --out_coef reports/calibrated_engine_weights_a10_f3.csv
```

### Step 3 — Verify output quickly
```bash
ls -lh reports | egrep "anchor_features_v2.csv|snr_engine_scores.csv|calibrated_engine_preds_a10_f3.csv|comparison_table.csv|baseline_v2"
head -n 5 reports/snr_engine_scores.csv
```

### Step 4 — Optional: create the comparison table artifact
```bash
python - <<'PY'
import pandas as pd

base = pd.read_csv("reports/baseline_v2/pred_a100_f3.csv").rename(
    columns={"snr_pred_trainfit": "snr_baseline_pred_trainfit"}
)
eng = pd.read_csv("reports/snr_engine_scores.csv")
cal = pd.read_csv("reports/calibrated_engine_preds_a10_f3.csv")

# keep compatibility if the calibrated column name differs
if "snr_engine_calibrated_pred_trainfit" not in cal.columns and "snr_calibrated_pred_trainfit" in cal.columns:
    cal = cal.rename(columns={"snr_calibrated_pred_trainfit":"snr_engine_calibrated_pred_trainfit"})

df = eng.merge(base[["video_id","snr_baseline_pred_trainfit"]], on="video_id", how="left") \
        .merge(cal[["video_id","snr_engine_calibrated_pred_trainfit"]], on="video_id", how="left")

out = df[[
    "video_id","title","snr_score",
    "snr_engine_score",
    "snr_baseline_pred_trainfit",
    "snr_engine_calibrated_pred_trainfit",
    "clarity_z","depth_z","penalty_z"
]]
out.to_csv("reports/comparison_table.csv", index=False)
print("Wrote: reports/comparison_table.csv")
print(out.head(3))
PY
```

---

## Quick demo: score **any** YouTube video (≤ 30 min)

This repo focuses on *feature extraction + scoring*. You can score any new video **if you can obtain the transcript text**.

### Step A — Download transcript (one-liner)

**Option 1: yt-dlp (recommended)**
```bash
# Install once (macOS): brew install yt-dlp
# Install once (pip): pip install yt-dlp

mkdir -p data/raw/youtube
yt-dlp --skip-download --write-auto-sub --sub-lang en --sub-format vtt \
  -o "data/raw/youtube/%(id)s.%(ext)s" \
  "<YOUTUBE_URL>"
```

Convert the `.vtt` to plain text:
```bash
python - <<'PY'
import glob, re, pathlib

vtts = sorted(glob.glob("data/raw/youtube/*.vtt"))
assert vtts, "No .vtt found under data/raw/youtube/. Did yt-dlp download captions?"
vtt_path = pathlib.Path(vtts[-1])

text = vtt_path.read_text(encoding="utf-8", errors="ignore")
# Remove timestamps and cue metadata; keep lines
lines = []
for ln in text.splitlines():
    ln = ln.strip()
    if not ln or ln.startswith("WEBVTT") or "-->" in ln:
        continue
    # drop speaker tags like "<v Speaker>"
    ln = re.sub(r"<[^>]+>", "", ln).strip()
    if ln:
        lines.append(ln)
out = "\n".join(lines)

out_path = vtt_path.with_suffix(".txt")
out_path.write_text(out, encoding="utf-8")
print("Wrote transcript:", out_path)
PY
```

### Step B — Create a minimal input CSV row in the same schema as the gold file

The feature builder expects the *same header schema* as `data/labels/gold_labels_llm_snrC.csv`.

Generate a template CSV with the correct columns:
```bash
python - <<'PY'
import pandas as pd
from pathlib import Path

gold = pd.read_csv("data/labels/gold_labels_llm_snrC.csv")
cols = list(gold.columns)

# create a 1-row template with empty values
tmpl = pd.DataFrame([{c: "" for c in cols}])

out = Path("data/labels/one_video_template.csv")
out.parent.mkdir(parents=True, exist_ok=True)
tmpl.to_csv(out, index=False)
print("Wrote template:", out)
print("Columns:", cols)
PY
```

Now open `data/labels/one_video_template.csv` and fill at least:
- `video_id` (YouTube id)
- `title`
- the transcript column (whatever name exists in your gold file — this prints the columns so you can see it)

> If your gold file uses a column like `transcript` / `raw_transcript` / similar, paste the transcript text there.

Rename it to:
```bash
mv data/labels/one_video_template.csv data/labels/one_video.csv
```

### Step C — Score the new video (same pipeline)
```bash
python scripts/build_anchor_features.py \
  --input data/labels/one_video.csv \
  --output reports/anchor_features_one_video.csv \
  --generic-advice-bank data/generic_advice_bank.txt

python scripts/compute_snr_engine.py \
  --input reports/anchor_features_one_video.csv \
  --output reports/snr_engine_one_video.csv

head -n 2 reports/snr_engine_one_video.csv
```

**Important:** The calibrated predictor is trained on the small gold seed set. For a brand-new video, the most honest output is the **engine components** (clarity/depth/penalty) and the raw engine score.

---

## Commands (match current scripts)

### 1) Build anchor features
```bash
python scripts/build_anchor_features.py \
  --input data/labels/gold_labels_llm_snrC.csv \
  --output reports/anchor_features_v2.csv \
  --generic-advice-bank data/generic_advice_bank.txt
```

### 2) Train baseline Ridge regression (direct-to-SNR)
```bash
python scripts/train_snr_baseline.py \
  --input reports/anchor_features_v2.csv \
  --alpha 100 \
  --folds 3 \
  --out_coef reports/baseline_v2/coef_a100_f3.csv \
  --out_pred reports/baseline_v2/pred_a100_f3.csv
```

### 3) Compute decomposed SNR engine scores
```bash
python scripts/compute_snr_engine.py \
  --input reports/anchor_features_v2.csv \
  --output reports/snr_engine_scores.csv
```

### 4) Calibrate the decomposed engine (3 weights + bias)
```bash
python scripts/calibrate_engine.py \
  --engine_csv reports/snr_engine_scores.csv \
  --alpha 10 \
  --folds 3 \
  --out_pred reports/calibrated_engine_preds_a10_f3.csv \
  --out_coef reports/calibrated_engine_weights_a10_f3.csv
```

---

## Results and design choice

### Baseline 1 — direct regression to `snr_score`
A direct Ridge regression baseline on anchor features is included as a sanity check, but it is **not expected to generalize well** with a tiny labeled dataset.

See: `reports/baseline_v2/BASELINE_SUMMARY.md`

### Key insight — define meaning explicitly; use ML only to calibrate
Instead of forcing a model to “invent” what signal means, the system decomposes SNR into:

- **Clarity proxy** (structure + evidence + density)
- **Depth proxy** (normalized entropy + diversity + sentence length)
- **Penalty score** (fear + promo + recycled-signal penalties)

A small Ridge model then calibrates these three interpretable components to the human rubric.

---

## License
MIT
