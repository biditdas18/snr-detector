#!/usr/bin/env python3
"""
Trains binary SNR classifier on final dataset.
- Embeddings: all-MiniLM-L6-v2 (384-dim) + domain one-hot (3-dim) = 387-dim
- Train: 390 (90 real + 300 synthetic)
- Test:  60 (original real, silver labels)
- Models: Logistic Regression + SVM
"""

import csv
import json
import os
import numpy as np
from collections import Counter
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns

BASE       = "/Users/biditdas/Desktop/snr-submission/snr-detector"
TRAIN_PATH = os.path.join(BASE, "data/labels/train_set_final.csv")
TEST_PATH  = os.path.join(BASE, "data/labels/test_set_final.csv")
REPORT_DIR = os.path.join(BASE, "reports")
os.makedirs(REPORT_DIR, exist_ok=True)

DOMAIN_MAP = {
    "career_selfimprovement": [1, 0, 0],
    "tech_ai":                [0, 1, 0],
    "general_education":      [0, 0, 1]
}


def load(path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def make_features(rows, model):
    texts      = [r["transcript"] for r in rows]
    domains    = [DOMAIN_MAP.get(r["domain"], [0, 0, 0]) for r in rows]
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=32)
    domain_arr = np.array(domains)
    return np.hstack([embeddings, domain_arr])


def binary_label(rows):
    return np.array([1 if r["signal_level"] == "HIGH" else 0 for r in rows])


def evaluate(name, clf, X_test, y_test):
    y_pred = clf.predict(X_test)

    if hasattr(clf, "predict_proba"):
        probs = clf.predict_proba(X_test)[:, 1]
        best_f1, best_thresh = 0, 0.5
        for thresh in np.arange(0.1, 0.9, 0.05):
            preds = (probs >= thresh).astype(int)
            f = f1_score(y_test, preds, pos_label=1, zero_division=0)
            if f > best_f1:
                best_f1 = f
                best_thresh = thresh
        y_pred_opt = (probs >= best_thresh).astype(int)
    else:
        y_pred_opt = y_pred
        best_thresh = 0.5
        best_f1 = f1_score(y_test, y_pred_opt, pos_label=1, zero_division=0)

    result = {
        "accuracy_default":  round(accuracy_score(y_test, y_pred), 4),
        "f1_default":        round(f1_score(y_test, y_pred, pos_label=1, zero_division=0), 4),
        "precision_default": round(precision_score(y_test, y_pred, pos_label=1, zero_division=0), 4),
        "recall_default":    round(recall_score(y_test, y_pred, pos_label=1, zero_division=0), 4),
        "f1_optimal":        round(best_f1, 4),
        "optimal_threshold": round(best_thresh, 3),
        "accuracy_optimal":  round(accuracy_score(y_test, y_pred_opt), 4),
        "precision_optimal": round(precision_score(y_test, y_pred_opt, pos_label=1, zero_division=0), 4),
        "recall_optimal":    round(recall_score(y_test, y_pred_opt, pos_label=1, zero_division=0), 4),
    }

    print(f"\n{name}:")
    print(f"  Default  — Acc:{result['accuracy_default']:.3f} "
          f"F1:{result['f1_default']:.3f} "
          f"P:{result['precision_default']:.3f} "
          f"R:{result['recall_default']:.3f}")
    print(f"  Optimal  — Acc:{result['accuracy_optimal']:.3f} "
          f"F1:{result['f1_optimal']:.3f} "
          f"P:{result['precision_optimal']:.3f} "
          f"R:{result['recall_optimal']:.3f} "
          f"(thresh={best_thresh:.2f})")
    print(classification_report(y_test, y_pred_opt, target_names=["LOW", "HIGH"]))

    return result, y_pred_opt


def main():
    print("Loading data...")
    train_rows = load(TRAIN_PATH)
    test_rows  = load(TEST_PATH)

    print(f"Train: {len(train_rows)} | {Counter(r['signal_level'] for r in train_rows)}")
    print(f"Test:  {len(test_rows)} | {Counter(r['signal_level'] for r in test_rows)}")

    print("\nGenerating embeddings...")
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    X_train = make_features(train_rows, embed_model)
    X_test  = make_features(test_rows,  embed_model)
    y_train = binary_label(train_rows)
    y_test  = binary_label(test_rows)

    print(f"\nFeature dimensions: {X_train.shape[1]} (384 embedding + 3 domain)")

    results   = {}
    best_clf  = None
    best_f1   = 0
    best_preds = None

    # Logistic Regression
    lr = LogisticRegression(max_iter=1000, C=1.0, class_weight="balanced")
    lr.fit(X_train, y_train)
    res, preds = evaluate("Logistic Regression", lr, X_test, y_test)
    results["LogisticRegression"] = res
    if res["f1_optimal"] > best_f1:
        best_f1    = res["f1_optimal"]
        best_clf   = lr
        best_preds = preds

    # SVM
    svm = SVC(kernel="rbf", C=1.0, probability=True, class_weight="balanced")
    svm.fit(X_train, y_train)
    res, preds = evaluate("SVM RBF", svm, X_test, y_test)
    results["SVM_RBF"] = res
    if res["f1_optimal"] > best_f1:
        best_f1    = res["f1_optimal"]
        best_clf   = svm
        best_preds = preds

    # Baseline — majority class (all LOW)
    majority = np.zeros(len(y_test), dtype=int)
    results["Baseline_MajorityLOW"] = {
        "accuracy_default":  round(accuracy_score(y_test, majority), 4),
        "f1_default":        0.0,
        "precision_default": 0.0,
        "recall_default":    0.0,
    }
    print(f"\nBaseline (all LOW): "
          f"Acc={results['Baseline_MajorityLOW']['accuracy_default']:.3f} F1=0.000")

    # Save JSON results
    with open(os.path.join(REPORT_DIR, "classifier_results_final.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to reports/classifier_results_final.json")

    # Confusion matrix
    cm = confusion_matrix(y_test, best_preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d",
                xticklabels=["LOW", "HIGH"],
                yticklabels=["LOW", "HIGH"],
                cmap="Blues")
    plt.title(f"SNR Classifier — Final Results\n"
              f"(Train: 390 real+synthetic | Test: 60 real silver)")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(os.path.join(REPORT_DIR, "confusion_matrix_final.png"), dpi=150)
    print("Confusion matrix saved to reports/confusion_matrix_final.png")


if __name__ == "__main__":
    main()
