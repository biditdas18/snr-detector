# SNR-Detector (v1)
**Signal-to-Noise Ratio Detection for Social Media Content**

A deterministic, interpretable system to distinguish **informative signal** from **high-confidence noise**
in social-media style content, using transcript-derived anchor features and a lightweight
calibration layer.

---

## Quick demo (2 minutes)

Run the end-to-end scorer on any YouTube video with captions:

```bash
python scripts/score_youtube.py "https://www.youtube.com/watch?v=UrcwDOEBzZE"
```

**Output includes:**
- SNR label: **HIGH / MID / LOW** (rule-based; most reliable in v1)
- SNR score: **1–5** (small-n calibrated)
- Raw engine score (uncalibrated)
- Component signals: clarity / depth / penalty
- Short natural-language explanation
- Diagnostic percentile (relative to calibration set)

> **Note**  
> Percentile is shown for diagnostics only.  
> Decisions are driven by the engine label and component signals.

---

## Why this exists

Modern social platforms reward:
- urgency over understanding
- repetition over novelty
- fear and promotion over clarity

This project scores content by **what it adds (signal)** versus **what it inflates (noise)**,
with a strong bias toward:
- reproducibility
- interpretability
- honest limitations

---

## What this is

- Deterministic transcript-based scoring (no stochastic inference)
- Decomposed SNR engine:
  - **Clarity** (structure, evidence, density)
  - **Depth** (lexical diversity, entropy proxies)
  - **Penalty** (fear, promo, recycled-signal patterns)
- Rule-based label with lightweight calibration
- Reproducible artifacts under the `reports/` directory

---

## What this is NOT

- Not a benchmark model
- Not trained on large datasets
- Not claiming statistical generalization
- Percentile is **not** a global ranking

This is an **exploratory v1** focused on interpretability and behavioral consistency.

---

## Setup

```bash
git clone https://github.com/biditdas18/snr-detector.git
cd snr-detector

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## How it works (high level)

1. Fetch or read transcript text  
2. Extract deterministic anchor features  
3. Compute decomposed SNR engine score  
4. Emit rule-based SNR label (**HIGH / MID / LOW**)  
5. Map to a calibrated **1–5** score (small-n)  
6. Display a diagnostic percentile vs calibration set  

---

## Testing on unseen videos

You can score **any** video with a transcript:

```bash
python scripts/score_youtube.py "<youtube_url>"
```

No tuning occurs during scoring.  
The system runs end-to-end on unseen content.

---

## Limitations

- Calibration set is small (~20–30 samples); scores may shrink toward the middle
- Percentile is relative to the calibration set only
- Transcript quality and domain affect results
- Long videos (>30 min) may reduce reliability in this prototype

---

## Research & reproducibility (optional)

This repository also includes:
- Anchor feature extraction pipeline
- Ridge-regression baseline for comparison
- Calibration experiments and analysis artifacts

These are provided for transparency and reproducibility, but are **not required**
to run the v1 scorer.

---

## License

MIT
