# SNR-Detect: Signal-to-Noise Ratio Detection for Educational Video Content

[![arXiv](https://img.shields.io/badge/arXiv-cs.CL-red)](https://arxiv.org/abs/ARXIV_ID_HERE)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

Binary classifier that detects whether a YouTube educational video
transcript is HIGH or LOW signal — based on content quality, not
engagement metrics.

**Best model:** SVM RBF — F1=0.871, Recall=0.964 on real YouTube transcripts.

---

## Results

| Model | Accuracy | F1 (HIGH) | Precision | Recall |
|-------|----------|-----------|-----------|--------|
| Baseline (majority LOW) | 0.533 | 0.000 | — | — |
| Logistic Regression | 0.850 | 0.857 | 0.771 | 0.964 |
| **SVM RBF** | **0.867** | **0.871** | **0.794** | **0.964** |

Train: 89 real + 300 synthetic transcripts
Test: 60 real transcripts (held out, never trained on)
Labels: LLM ensemble majority vote (GPT-4o mini + Gemini 2.5 Flash + Llama 3.3 70B)
Agreement: 74% full consensus, 82% avg pairwise (up from 53% baseline)

---

## SNR Taxonomy

| Label | What it means |
|-------|--------------|
| **HIGH** | Concrete actionable steps, named frameworks, methodological coherence, solution-oriented |
| **LOW** | Generic advice, fear-based framing, heavy promotion, repetition without progression |

Domain-specific rubrics were calibrated via iterative multi-agent agreement
optimization. General education: 96.7% weighted agreement. Tech/AI: 81.7%.
Career: 70% (career self-improvement is harder to calibrate across model architectures).

---

## Dataset

| Split | Source | Size | Labels |
|-------|--------|------|--------|
| Test | Real YouTube (held out) | 60 | LLM ensemble |
| Train | Real YouTube | 89 | LLM ensemble |
| Train | Synthetic (Claude-generated) | 300 | By construction |
| **Total** | | **449** | |

Domains: Career & Self-Improvement · Technology & AI · General Education

---

## Repository Structure

```
snr-detector/
├── data/
│   ├── labels/
│   │   ├── all_150_silver_labels.csv    # All 150 real transcripts + labels
│   │   ├── test_set_final.csv           # 60 held-out test transcripts
│   │   └── train_set_final.csv          # 389 training transcripts
│   ├── synthetic/
│   │   └── synthetic_transcripts.csv    # 300 synthetic training transcripts
│   └── transcripts_new/                 # 90 new real transcripts (JSONL)
├── experiments/
│   └── train_final_classifier.py        # Main training + evaluation script
├── scripts/
│   ├── label_150_transcripts.py         # LLM ensemble labeling (needs API keys)
│   ├── build_final_datasets.py          # Build train/test splits
│   └── fetch_new_transcripts.py         # Collect YouTube transcripts
└── reports/
    ├── classifier_results_final.json    # Full results JSON
    └── confusion_matrix_final.png       # Confusion matrix
```

---

## Quickstart

### 1. Install dependencies

```bash
git clone https://github.com/biditdas18/snr-detector.git
cd snr-detector
pip install -r requirements.txt
```

### 2. Train and evaluate classifier

```bash
python experiments/train_final_classifier.py
```

This trains on `data/labels/train_set_final.csv` and evaluates on
`data/labels/test_set_final.csv`. No API keys needed — data is
already labeled and included.

Expected output:

```
SVM RBF:
  Accuracy: 0.867 | F1: 0.871 | Precision: 0.794 | Recall: 0.964
```

### 3. Re-label transcripts (optional — requires API keys)

If you want to re-run the full labeling pipeline:

```bash
export OPENAI_API_KEY="your-key"
export GOOGLE_API_KEY="your-key"
export GROQ_API_KEY="your-key"

python scripts/label_150_transcripts.py
python scripts/build_final_datasets.py
python experiments/train_final_classifier.py
```

**Never commit API keys to the repository.**

---

## Deployment Vision

**Personal tool:** User bias is a feature. The classifier learns your
definition of educational value from your feedback history.

**Platform tool:** Population-level consensus washes out individual bias.
YouTube's "was this helpful" signal is an untapped training source.

The classifier runs in milliseconds on CPU — no API cost at inference time.

---

## Limitations

- Silver labels from LLM ensemble (human validation planned)
- 90 real training transcripts from established educational channels
  (skews HIGH — LOW signal from synthetic data)
- YouTube only (shorter-form platforms require adaptation)
- Three domains (medical, legal, financial not yet covered)

---

## Citation

```bibtex
@article{das2025snr,
  title={Signal-to-Noise Ratio Detection in Educational Video Content:
         A Binary Classification Framework Using Transcript Embeddings},
  author={Das, Bidit},
  journal={arXiv preprint arXiv:ARXIV_ID_HERE},
  year={2025}
}
```

---

## Author

**Bidit Das** — Independent Researcher · AWS Escalation Engineer · Dallas, TX
- GitHub: [@biditdas18](https://github.com/biditdas18)
- Medium: [@biditdas18](https://medium.com/@biditdas18)

*Replace `ARXIV_ID_HERE` with actual arXiv ID after submission.*
