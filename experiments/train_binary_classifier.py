import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, f1_score,
    precision_score, recall_score, confusion_matrix,
    classification_report)
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

results = {}

for name, clf in [
    ("Logistic Regression", LogisticRegression(max_iter=1000, C=1.0)),
    ("SVM RBF",             SVC(kernel="rbf", C=1.0, probability=True))
]:
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    results[name] = {
        "accuracy":  round(accuracy_score(y_test, y_pred), 4),
        "f1":        round(f1_score(y_test, y_pred, pos_label=1, zero_division=0), 4),
        "precision": round(precision_score(y_test, y_pred, pos_label=1, zero_division=0), 4),
        "recall":    round(recall_score(y_test, y_pred, pos_label=1, zero_division=0), 4),
    }
    print(f"\n{name}:")
    for k, v in results[name].items():
        print(f"  {k}: {v}")
    print(classification_report(y_test, y_pred, target_names=["LOW","HIGH"]))

# V1 baseline — majority class (all LOW)
y_majority = np.zeros(len(y_test), dtype=int)
results["V1 Baseline (majority)"] = {
    "accuracy":  round(accuracy_score(y_test, y_majority), 4),
    "f1":        round(f1_score(y_test, y_majority, pos_label=1, zero_division=0), 4),
    "precision": round(precision_score(y_test, y_majority, pos_label=1, zero_division=0), 4),
    "recall":    round(recall_score(y_test, y_majority, pos_label=1, zero_division=0), 4),
}
print(f"\nV1 Baseline (majority class): {results['V1 Baseline (majority)']}")

# Save results
with open("reports/classifier_results.json", "w") as f:
    json.dump(results, f, indent=2)

# Confusion matrix for best model
best_clf = LogisticRegression(max_iter=1000, C=1.0)
best_clf.fit(X_train, y_train)
y_pred_best = best_clf.predict(X_test)
cm = confusion_matrix(y_test, y_pred_best)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d",
            xticklabels=["LOW","HIGH"],
            yticklabels=["LOW","HIGH"],
            cmap="Blues")
plt.title("SNR Binary Classifier — Confusion Matrix\n(Trained: Synthetic | Evaluated: Real)")
plt.ylabel("True Label")
plt.xlabel("Predicted Label")
plt.tight_layout()
plt.savefig("reports/confusion_matrix.png", dpi=150)
print("\nSaved: reports/confusion_matrix.png")
print("\nAll results saved to reports/classifier_results.json")
