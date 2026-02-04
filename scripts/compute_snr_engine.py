#!/usr/bin/env python3
import argparse
import numpy as np
import pandas as pd


EPS = 1e-9


def z_with_ref(x: pd.Series, ref: pd.Series) -> pd.Series:
    """
    Z-score x using mean/std computed from ref.
    Works for single-row x as long as ref has variance.
    """
    mu = float(ref.mean())
    sigma = float(ref.std(ddof=0))
    sigma = sigma if sigma > EPS else EPS
    return (x - mu) / sigma


def z_within(x: pd.Series) -> pd.Series:
    """Z-score within the provided series (batch-normalization)."""
    return (x - x.mean()) / (x.std(ddof=0) + EPS)


def compute_clarity(df: pd.DataFrame) -> pd.Series:
    return (
        0.4 * df["structure_hits"] +
        0.4 * df["evidence_hits"] +
        0.2 * df["content_density_proxy"]
    )


def compute_depth(df: pd.DataFrame) -> pd.Series:
    # Normalize entropy by length to avoid "long transcript = deep"
    entropy_norm = np.log1p(df["entropy"] / (df["word_count"] + EPS))
    avg_sent = np.log1p(df["avg_sent_len"])
    uniq = df["unique_ratio"]
    return 0.5 * entropy_norm + 0.3 * uniq + 0.2 * avg_sent


def compute_penalty(df: pd.DataFrame) -> pd.Series:
    return (
        0.5 * df["fear_hits"] +
        0.3 * df["promo_hits"] +
        0.2 * df["recycled_penalty"]
    )


def validate_required(df: pd.DataFrame, required: list[str], name: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"{name} is missing required columns:\n" + "\n".join(f"- {c}" for c in missing)
        )


def add_proxies(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["clarity_proxy"] = compute_clarity(df)
    df["depth_proxy"] = compute_depth(df)
    df["penalty_score"] = compute_penalty(df)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Anchor features CSV to score")
    ap.add_argument("--output", required=True, help="Output CSV with engine scores")
    ap.add_argument(
        "--ref_csv",
        default=None,
        help=(
            "Optional reference anchor CSV used to compute z-scores (recommended for single-video scoring). "
            "If provided, z-scores are computed against this reference distribution."
        ),
    )
    args = ap.parse_args()

    df = pd.read_csv(args.input)

    required = [
        "video_id", "title",
        "word_count", "entropy", "unique_ratio", "avg_sent_len",
        "structure_hits", "evidence_hits", "content_density_proxy",
        "fear_hits", "promo_hits", "recycled_penalty",
    ]
    # snr_score is not required for scoring arbitrary videos; include if available.
    validate_required(df, required, "Input CSV")

    # ---- 1) Compute proxies (raw) ----
    df = add_proxies(df)

    # ---- 2) Compute z-scores ----
    used_ref = False
    if args.ref_csv:
        ref = pd.read_csv(args.ref_csv)
        validate_required(ref, required, "Reference CSV")
        ref = add_proxies(ref)

        # If ref has no variance in a proxy, z would be degenerate; still safe due to EPS.
        df["clarity_z"] = z_with_ref(df["clarity_proxy"], ref["clarity_proxy"])
        df["depth_z"] = z_with_ref(df["depth_proxy"], ref["depth_proxy"])
        df["penalty_z"] = z_with_ref(df["penalty_score"], ref["penalty_score"])
        used_ref = True
    else:
        # Within-batch z-scoring (works only when scoring multiple rows together)
        df["clarity_z"] = z_within(df["clarity_proxy"])
        df["depth_z"] = z_within(df["depth_proxy"])
        df["penalty_z"] = z_within(df["penalty_score"])

    # ---- 3) Interpretable engine score ----
    df["snr_engine_score"] = 0.5 * df["clarity_z"] + 0.5 * df["depth_z"] - df["penalty_z"]

    keep_cols = [
        "video_id", "title",
        "clarity_proxy", "depth_proxy", "penalty_score",
        "clarity_z", "depth_z", "penalty_z",
        "snr_engine_score",
    ]

    # If snr_score exists (gold set), keep it for correlation/debug
    if "snr_score" in df.columns:
        keep_cols.insert(2, "snr_score")

    out = df[keep_cols].copy()
    out.to_csv(args.output, index=False)

    msg = f"Wrote SNR engine scores to: {args.output}"
    if used_ref:
        msg += f"  (z-scores referenced to: {args.ref_csv})"
    else:
        msg += "  (z-scores computed within input batch)"
    print(msg)


if __name__ == "__main__":
    main()
