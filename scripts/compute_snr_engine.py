#!/usr/bin/env python3
import argparse
import numpy as np
import pandas as pd


def z(x: pd.Series) -> pd.Series:
    return (x - x.mean()) / (x.std(ddof=0) + 1e-9)


def compute_clarity(df: pd.DataFrame) -> pd.Series:
    return (
        0.4 * df["structure_hits"] +
        0.4 * df["evidence_hits"] +
        0.2 * df["content_density_proxy"]
    )


def compute_depth(df: pd.DataFrame) -> pd.Series:
    # Normalize entropy by length to avoid "long transcript = deep"
    entropy_norm = np.log1p(df["entropy"] / (df["word_count"] + 1e-9))
    avg_sent = np.log1p(df["avg_sent_len"])
    uniq = df["unique_ratio"]
    return 0.5 * entropy_norm + 0.3 * uniq + 0.2 * avg_sent


def compute_penalty(df: pd.DataFrame) -> pd.Series:
    return (
        0.5 * df["fear_hits"] +
        0.3 * df["promo_hits"] +
        0.2 * df["recycled_penalty"]
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Anchor features CSV")
    ap.add_argument("--output", required=True, help="Output CSV with SNR engine scores")
    args = ap.parse_args()

    df = pd.read_csv(args.input)

    required = [
        "video_id", "title", "snr_score",
        "word_count", "entropy", "unique_ratio", "avg_sent_len",
        "structure_hits", "evidence_hits", "content_density_proxy",
        "fear_hits", "promo_hits", "recycled_penalty",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError("Missing required columns:\n" + "\n".join(f"- {c}" for c in missing))

    # ---- 1) Compute proxies (must exist before normalization) ----
    df["clarity_proxy"] = compute_clarity(df)
    df["depth_proxy"] = compute_depth(df)
    df["penalty_score"] = compute_penalty(df)

    # ---- 2) Normalize proxies for stable combination ----
    df["clarity_z"] = z(df["clarity_proxy"])
    df["depth_z"] = z(df["depth_proxy"])
    df["penalty_z"] = z(df["penalty_score"])

    # ---- 3) Human-defined engine score (interpretable) ----
    df["snr_engine_score"] = 0.5 * df["clarity_z"] + 0.5 * df["depth_z"] - df["penalty_z"]

    keep_cols = [
        "video_id", "title", "snr_score",
        "clarity_proxy", "depth_proxy", "penalty_score",
        "clarity_z", "depth_z", "penalty_z",
        "snr_engine_score",
    ]
    df[keep_cols].to_csv(args.output, index=False)
    print(f"Wrote SNR engine scores to: {args.output}")


if __name__ == "__main__":
    main()
