# SNR-Detector  
**Signal-to-Noise Ratio Detection for Social Media Content**

> A deterministic, interpretable system to distinguish *informative signal* from *high-confidence noise* in short-form online content.

---

## Motivation

Modern social media rewards:
- urgency over accuracy  
- repetition over novelty  
- fear over clarity  

This project introduces a **Signal-to-Noise Ratio (SNR)** framework that scores content based on *what it adds*, not *how loud it sounds*.

The goal is **human-interpretable detection**, not black-box virality prediction.

---

## Core Principles

1. **Interpretability First**  
   Every score must be explainable in plain language.

2. **Deterministic Anchors**  
   Features behave like “physics laws” — reproducible and stable.

3. **Reproducibility**  
   A non-ML engineer should be able to rerun the pipeline end-to-end.

4. **Paper-Grade Output**  
   Results must be strong enough for IEEE-style submission.

---

## What This Is (and Is Not)

### ✅ This is
- A content **quality diagnostic**
- A **human-aligned filtering** system
- A research-grade ML + rules hybrid

### ❌ This is not
- A recommendation engine
- A sentiment analyzer
- A censorship or moderation tool

---

## High-Level Architecture

```
Video / Text
   ↓
Transcript Extraction
   ↓
Anchor Feature Generation
   ↓
Baseline ML Model (Ridge)
   ↓
Explainability + Actionability
   ↓
Final SNR Score
```

---

## Repository Structure

```
snr-detector/
│
├── data/
│   ├── raw/                    # Raw transcripts
│   ├── labels/                 # Gold seed labels
│   └── processed/
│
├── scripts/
│   ├── build_anchor_features.py
│   ├── train_snr_baseline.py
│   ├── evaluate.py
│   └── utils.py
│
├── reports/
│   ├── anchor_features_v2.csv
│   ├── baseline_v2/
│   └── figures/
│
├── notebooks/
│   └── analysis.ipynb
│
├── README.md
└── requirements.txt
```

---

## Anchor Features (Physics Layer)

Anchor features are deterministic signals extracted directly from content.

Examples:
- Informational density  
- Redundancy / recycled signal similarity  
- Emotional urgency inflation  
- Actionability score  
- Explainability score  

These features:
- do not depend on labels  
- can be recomputed on any transcript  
- are reusable across models  

---

## Gold Seed Dataset

The gold dataset is **intentionally small and high-quality**.

Labeling logic:
- **Signal** = clarity + novelty + actionable insight  
- **Noise** = fear, promo, recycled advice, urgency inflation  

Each sample includes:
- raw transcript  
- explainability score  
- actionability score  
- final human-assigned SNR  

---

## Feature Generation

```bash
python scripts/build_anchor_features.py   --input data/labels/gold_labels_llm_snrC.csv   --output reports/anchor_features_v2.csv
```

---

## Model Training (Baseline)

```bash
python scripts/train_snr_baseline.py   --input reports/anchor_features_v2.csv   --labels data/labels/gold_labels_llm_snrC.csv   --outdir reports/baseline_v2
```

---

## Evaluation

Evaluation focuses on:
- rank consistency  
- error distribution  
- failure modes (fear vs recycled signal)  

---

## Reproducibility

```bash
pip install -r requirements.txt
python scripts/build_anchor_features.py
python scripts/train_snr_baseline.py
```

---

## Research Direction

Planned extensions:
- distinct **Recycled Signal** class  
- temporal novelty decay  
- cross-platform robustness tests  

---

## Ethical Stance

This project explicitly rejects:
- engagement-maximization  
- fear-based amplification  
- opaque scoring  

---

## License

MIT License  

---

## Author

Independent research project  
Built with a long-term goal of **human-aligned machine intelligence**.
