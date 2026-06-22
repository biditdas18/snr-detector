"""
Ablation runs for arXiv submission cleanup.
- Task 7: synthetic-only training, evaluated on test_set_final.csv
- Task 8: no-domain-feature training (384-dim only), evaluated on test_set_final.csv
Both use the same feature pipeline as the production run (all-MiniLM-L6-v2 + optional 3-dim domain one-hot).
"""
import json
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

DOMAIN_MAP = {"career_selfimprovement": 0, "tech_ai": 1, "general_education": 2}

def domain_onehot(df):
    ohe = np.zeros((len(df), 3))
    for i, d in enumerate(df["domain"]):
        idx = DOMAIN_MAP.get(d, -1)
        if idx >= 0:
            ohe[i, idx] = 1.0
    return ohe

def metrics(y_true, y_pred):
    return {
        "accuracy":  round(float(accuracy_score(y_true, y_pred)), 4),
        "f1":        round(float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)), 4),
        "precision": round(float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)), 4),
        "recall":    round(float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)), 4),
    }

print("Loading data...")
synth = pd.read_csv("data/synthetic/synthetic_transcripts.csv")
test  = pd.read_csv("data/labels/test_set_final.csv")

print(f"Synthetic train: {len(synth)} | Test: {len(test)}")

print("Embedding transcripts (this takes ~30s)...")
model = SentenceTransformer("all-MiniLM-L6-v2")
X_synth_emb = model.encode(synth["transcript"].tolist(), show_progress_bar=True)
X_test_emb  = model.encode(test["transcript"].tolist(),  show_progress_bar=True)

y_synth = (synth["signal_level"] == "HIGH").astype(int).values
y_test  = (test["signal_level"]  == "HIGH").astype(int).values

synth_ohe = domain_onehot(synth)
test_ohe  = domain_onehot(test)

# 387-dim (with domain)
X_synth_387 = np.hstack([X_synth_emb, synth_ohe])
X_test_387  = np.hstack([X_test_emb, test_ohe])

# 384-dim (no domain)
X_synth_384 = X_synth_emb
X_test_384  = X_test_emb

# ── TASK 7: Synthetic-only, 387-dim ─────────────────────────────────────────
print("\n=== TASK 7: Synthetic-only baseline (387-dim) ===")

svm_synth = SVC(kernel="rbf", C=1.0, class_weight="balanced", probability=True)
svm_synth.fit(X_synth_387, y_synth)
y_pred_svm = (svm_synth.predict_proba(X_test_387)[:, 1] >= 0.5).astype(int)
svm_synth_metrics = metrics(y_test, y_pred_svm)
print(f"SVM RBF (synthetic-only, threshold=0.5): {svm_synth_metrics}")

lr_synth = LogisticRegression(max_iter=1000, C=1.0, class_weight="balanced")
lr_synth.fit(X_synth_387, y_synth)
y_pred_lr = (lr_synth.predict_proba(X_test_387)[:, 1] >= 0.5).astype(int)
lr_synth_metrics = metrics(y_test, y_pred_lr)
print(f"LR (synthetic-only, threshold=0.5):      {lr_synth_metrics}")

task7 = {"SVM_RBF_synthetic_only": svm_synth_metrics, "LR_synthetic_only": lr_synth_metrics}
with open("reports/classifier_results_synthetic_only_finaltest.json", "w") as f:
    json.dump(task7, f, indent=2)
print("Saved: reports/classifier_results_synthetic_only_finaltest.json")

# ── TASK 8: No domain feature, SVM (full train_set_final) ───────────────────
print("\n=== TASK 8: No-domain ablation (384-dim, full train) ===")
train = pd.read_csv("data/labels/train_set_final.csv")
print(f"Full train: {len(train)}")
X_train_emb = model.encode(train["transcript"].tolist(), show_progress_bar=True)
y_train = (train["signal_level"] == "HIGH").astype(int).values

svm_nodomain = SVC(kernel="rbf", C=1.0, class_weight="balanced", probability=True)
svm_nodomain.fit(X_train_emb, y_train)
y_pred_nd = (svm_nodomain.predict_proba(X_test_384)[:, 1] >= 0.5).astype(int)
nodomain_metrics = metrics(y_test, y_pred_nd)
print(f"SVM RBF (no domain feature, threshold=0.5): {nodomain_metrics}")

task8 = {"SVM_RBF_no_domain_feature": nodomain_metrics}
with open("reports/ablation_no_domain.json", "w") as f:
    json.dump(task8, f, indent=2)
print("Saved: reports/ablation_no_domain.json")

print("\n=== SUMMARY ===")
print(f"Task 7 — Synthetic-only SVM:    F1={svm_synth_metrics['f1']}  Acc={svm_synth_metrics['accuracy']}")
print(f"Task 7 — Synthetic-only LR:     F1={lr_synth_metrics['f1']}   Acc={lr_synth_metrics['accuracy']}")
print(f"Task 8 — No-domain SVM:         F1={nodomain_metrics['f1']}  Acc={nodomain_metrics['accuracy']}")
print(f"Production SVM (387-dim, mixed): F1=0.871 Acc=0.867  (from classifier_results_final.json)")
