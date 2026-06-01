#!/usr/bin/env python3
"""
Builds final train and test sets from labeled data.

Test set:  60 original transcripts (new consistent labels)
Train set: 90 new real (silver labels) + 300 synthetic (by construction)
           shuffled together

Output:
  data/labels/test_set_final.csv    (60 rows)
  data/labels/train_set_final.csv   (390 rows)
"""

import csv
import os
import random
from collections import Counter

BASE       = "/Users/biditdas/Desktop/snr-submission/snr-detector"
ALL_LABELS = os.path.join(BASE, "data/labels/all_150_silver_labels.csv")
SYNTHETIC  = os.path.join(BASE, "data/synthetic/synthetic_transcripts.csv")
TEST_OUT   = os.path.join(BASE, "data/labels/test_set_final.csv")
TRAIN_OUT  = os.path.join(BASE, "data/labels/train_set_final.csv")


def load(path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def save(rows, path, fields):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"Saved {len(rows)} rows → {path}")


def standardize(rows, source):
    return [{
        "domain":       r["domain"],
        "transcript":   r["transcript"],
        "signal_level": r["signal_level"],
        "source":       source
    } for r in rows]


def main():
    all_labeled = load(ALL_LABELS)
    synthetic   = load(SYNTHETIC)

    original_60 = [r for r in all_labeled if r["source"] == "original_60"]
    new_90      = [r for r in all_labeled if r["source"] == "new_90"]

    print(f"Original 60: {len(original_60)} | "
          f"{Counter(r['signal_level'] for r in original_60)}")
    print(f"New 90:      {len(new_90)} | "
          f"{Counter(r['signal_level'] for r in new_90)}")
    print(f"Synthetic:   {len(synthetic)} | "
          f"{Counter(r['signal_level'] for r in synthetic)}")

    fields = ["domain", "transcript", "signal_level", "source"]

    test = standardize(original_60, "original_silver")
    save(test, TEST_OUT, fields)

    train = standardize(new_90, "new_silver") + \
            standardize(synthetic, "synthetic")
    random.seed(42)
    random.shuffle(train)
    save(train, TRAIN_OUT, fields)

    print(f"\nTrain dist: {Counter(r['signal_level'] for r in train)}")
    print(f"Test dist:  {Counter(r['signal_level'] for r in test)}")


if __name__ == "__main__":
    main()
