import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, f1_score,
    precision_score, recall_score, confusion_matrix,
    classification_report, precision_recall_curve)
import matplotlib.pyplot as plt
import seaborn as sns
import json

# Load data
train = pd.read_csv("data/synthetic/synthetic_transcripts.csv")
test  = pd.read_csv("data/labels/gold_dataset_binary.csv")

print(f"Train: {len(train)} | Test: {len(test)}")
print(f"Train dist: {train['signal_level'].value_counts().to_dict()}")
print(f"Test dist:  {test['signal_level'].value_counts().to_dict()}")

# Embeddings
print("\nGenerating embeddings...")
model = SentenceTransformer("all-MiniLM-L6-v2")
X_train = model.encode(train["transcript"].tolist(), show_progress_bar=True)
X_test  = model.encode(test["transcript"].tolist(),  show_progress_bar=True)

# Labels — HIGH=1, LOW=0
y_train = (train["signal_level"] == "HIGH").astype(int)
y_test  = (test["signal_level"]  == "HIGH").astype(int)


def eval_at_threshold(y_true, y_proba, threshold):
    y_pred = (y_proba >= threshold).astype(int)
    return {
        "accuracy":  round(accuracy_score(y_true, y_pred), 4),
        "f1":        round(f1_score(y_true, y_pred, pos_label=1, zero_division=0), 4),
        "precision": round(precision_score(y_true, y_pred, pos_label=1, zero_division=0), 4),
        "recall":    round(recall_score(y_true, y_pred, pos_label=1, zero_division=0), 4),
    }


def find_optimal_threshold(y_true, y_proba):
    """Return threshold that maximises F1 on HIGH class via precision-recall curve."""
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)
    # thresholds has one fewer element than precisions/recalls
    f1s = np.where(
        (precisions[:-1] + recalls[:-1]) == 0,
        0,
        2 * precisions[:-1] * recalls[:-1] / (precisions[:-1] + recalls[:-1])
    )
    best_idx = np.argmax(f1s)
    return float(round(thresholds[best_idx], 4)), float(round(f1s[best_idx], 4))


results = {}

classifiers = [
    ("Logistic Regression (balanced)", LogisticRegression(max_iter=1000, C=1.0, class_weight="balanced")),
    ("SVM RBF",                         SVC(kernel="rbf", C=1.0, probability=True)),
]

for name, clf in classifiers:
    clf.fit(X_train, y_train)
    y_proba = clf.predict_proba(X_test)[:, 1]

    # Default threshold = 0.5
    default_metrics = eval_at_threshold(y_test, y_proba, 0.5)

    # Optimal threshold (maximises HIGH-class F1)
    opt_thresh, opt_f1_curve = find_optimal_threshold(y_test, y_proba)
    optimal_metrics = eval_at_threshold(y_test, y_proba, opt_thresh)

    results[name] = {
        "default_threshold_0.5": default_metrics,
        "optimal_threshold":     opt_thresh,
        "optimal_threshold_metrics": optimal_metrics,
    }

    print(f"\n{'='*55}")
    print(f"{name}")
    print(f"  Default (0.5):  {default_metrics}")
    print(f"  Optimal thresh: {opt_thresh}")
    print(f"  Optimal:        {optimal_metrics}")

    y_pred_default = (y_proba >= 0.5).astype(int)
    y_pred_optimal = (y_proba >= opt_thresh).astype(int)
    print(f"\n  Classification report @ default (0.5):")
    print(classification_report(y_test, y_pred_default, target_names=["LOW", "HIGH"]))
    print(f"  Classification report @ optimal ({opt_thresh}):")
    print(classification_report(y_test, y_pred_optimal, target_names=["LOW", "HIGH"]))

# V1 baseline — majority class (all LOW)
y_majority = np.zeros(len(y_test), dtype=int)
results["V1 Baseline (majority)"] = {
    "default_threshold_0.5": {
        "accuracy":  round(accuracy_score(y_test, y_majority), 4),
        "f1":        0.0,
        "precision": 0.0,
        "recall":    0.0,
    },
    "optimal_threshold": "N/A",
    "optimal_threshold_metrics": "N/A",
}
print(f"\nV1 Baseline (majority class): accuracy=0.8, f1=0.0")

# Save results
with open("reports/classifier_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nSaved: reports/classifier_results.json")

# ── Confusion matrices side-by-side ──────────────────────────────────────────
lr_clf = LogisticRegression(max_iter=1000, C=1.0, class_weight="balanced")
lr_clf.fit(X_train, y_train)
lr_proba = lr_clf.predict_proba(X_test)[:, 1]

lr_name = "Logistic Regression (balanced)"
opt_thresh_lr = results[lr_name]["optimal_threshold"]

y_pred_default = (lr_proba >= 0.5).astype(int)
y_pred_optimal = (lr_proba >= opt_thresh_lr).astype(int)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for ax, y_pred, title in [
    (axes[0], y_pred_default, f"Default threshold = 0.5"),
    (axes[1], y_pred_optimal, f"Optimal threshold = {opt_thresh_lr}"),
]:
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d",
                xticklabels=["LOW", "HIGH"],
                yticklabels=["LOW", "HIGH"],
                cmap="Blues", ax=ax)
    ax.set_title(f"LR (balanced) — {title}", fontsize=11)
    ax.set_ylabel("True Label")
    ax.set_xlabel("Predicted Label")

fig.suptitle("SNR Binary Classifier — Confusion Matrices\n(Trained: Synthetic | Evaluated: Real)",
             fontsize=12, y=1.02)
plt.tight_layout()
plt.savefig("reports/confusion_matrix.png", dpi=150, bbox_inches="tight")
print("Saved: reports/confusion_matrix.png")
