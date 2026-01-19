#!/usr/bin/env python3
import argparse, json
from pathlib import Path
import sys
import subprocess
import pandas as pd

from src.transcript_fetcher import extract_video_id, fetch_title_with_ytdlp, fetch_transcript
from src.preprocess import featurize_transcript_row
from src.snr_classifier import load_or_train_model, predict_score, explain_prediction

REPORTS = Path("reports")
MODELS  = Path("models")
REPORTS.mkdir(exist_ok=True)
MODELS.mkdir(exist_ok=True)

LATEST_JSON = REPORTS / "latest_result.json"
LATEST_CSV  = REPORTS / "latest_result.csv"

THRESH_NOISE_LT = 2.8
THRESH_HIGH_GE  = 3.6


def verdict(score: float) -> str:
    if score < THRESH_NOISE_LT:
        return "NOISE"
    if score < THRESH_HIGH_GE:
        return "MEDIUM_SIGNAL"
    return "HIGH_SIGNAL"


def ensure_anchor_features(
    out_csv: Path = REPORTS / "anchor_features_v2.csv",
    gold_labels_csv: Path = Path("data/labels/gold_labels_llm_snrC.csv"),
    build_script: Path = Path("scripts/build_anchor_features.py"),
) -> Path:
    """
    Ensures reports/anchor_features_v2.csv exists. If missing, build it from the gold labels CSV
    using scripts/build_anchor_features.py so fresh clones can train reproducibly.
    """
    if out_csv.exists():
        return out_csv

    if not gold_labels_csv.exists():
        raise SystemExit(
            f"Missing gold labels file: {gold_labels_csv}. "
            "This repo must include the gold labels CSV to train from source."
        )

    if not build_script.exists():
        raise SystemExit(
            f"Missing feature builder script: {build_script}. "
            "Expected scripts/build_anchor_features.py to exist."
        )

    out_csv.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, str(build_script),
        "--input", str(gold_labels_csv),
        "--output", str(out_csv),
    ]
    print("Building anchor features:")
    print("  " + " ".join(cmd))
    subprocess.check_call(cmd)
    return out_csv


def cmd_score(args):
    url = args.url.strip()
    vid = extract_video_id(url)
    if not vid:
        raise SystemExit("Could not parse video_id from the provided URL.")

    title = fetch_title_with_ytdlp(url) or ""
    transcript = fetch_transcript(vid)
    if not transcript:
        raise SystemExit("No transcript found (captions disabled/unavailable). Try another video.")

    # Build 1-row feature DF (your existing helper)
    X_eval = featurize_transcript_row(
        video_id=vid,
        title=title,
        transcript=transcript,
        generic_advice_bank_path=Path("data/generic_advice_bank.txt"),
        build_script_path=Path("scripts/build_anchor_features.py"),
    )

    model_path = MODELS / "snr_ridge.joblib"
    train_csv  = ensure_anchor_features()  # <-- build if missing

    model = load_or_train_model(train_csv=train_csv, model_path=model_path)

    score = float(predict_score(model, X_eval))
    score_r = round(score, 3)
    v = verdict(score)

    reasons = explain_prediction(model, X_eval, top_k_pos=3, top_k_neg=2)

    out_row = {"video_id": vid, "title": title, "snr_pred": score_r, "verdict": v}
    pd.DataFrame([out_row]).to_csv(LATEST_CSV, index=False)

    payload = {
        **out_row,
        "thresholds": {"noise_lt": THRESH_NOISE_LT, "high_ge": THRESH_HIGH_GE},
        "reasons": reasons,
    }
    LATEST_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("SNR Detector v1")
    print(f'Video: "{title}" ({vid})')
    print(f"SNR score (1–5): {score_r}")
    print(f"Verdict: {v}\n")
    print("Top reasons:")
    for r in reasons:
        print(f"- {r}")
    print(f"\nArtifacts saved:\n- {LATEST_JSON}\n- {LATEST_CSV}")


def cmd_train(args):
    from src.snr_classifier import train_and_save

    model_path = MODELS / "snr_ridge.joblib"
    model_path.parent.mkdir(parents=True, exist_ok=True)

    train_csv = ensure_anchor_features()  # <-- build if missing
    train_and_save(train_csv, model_path)

    print(f"Saved model: {model_path}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("score", help="Score a YouTube video link")
    s.add_argument("url")
    s.set_defaults(func=cmd_score)

    t = sub.add_parser("train", help="Train model from gold labels (auto-build anchor features)")
    t.set_defaults(func=cmd_train)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
