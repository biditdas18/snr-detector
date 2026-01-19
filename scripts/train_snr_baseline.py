#!/usr/bin/env python3
import argparse
import numpy as np
import pandas as pd

from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# Explicit allowlist of true anchor features (no labels / no rubric / no text)
ANCHOR_FEATURES = [
    "word_count",
    "sent_count",
    "avg_sent_len",
    "unique_ratio",
    "entropy",
    "top_trigram_rep",
    "recycled_signal_similarity",
    "recycled_penalty",
    "fear_hits",
    "promo_hits",
    "hype_hits",
    "generic_advice_hits",
    "evidence_hits",
    "structure_hits",
    "qmark_count",
    "exclam_count",
    "you_count",
    "i_count",
    "modal_count",
    "imperative_count",
    "content_density_proxy",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to anchor_features.csv")
    ap.add_argument("--out_coef", default="reports/snr_baseline_coeffs.csv")
    ap.add_argument("--out_pred", default="reports/snr_baseline_train_preds.csv")
    ap.add_argument("--target", default="snr_score")
    ap.add_argument("--alpha", type=float, default=1.0)  # Ridge strength
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()

    df = pd.read_csv(args.input)

    if args.target not in df.columns:
        raise ValueError(f"--target '{args.target}' not found in input columns.")

    # Verify all required feature columns exist
    missing = [c for c in ANCHOR_FEATURES if c not in df.columns]
    if missing:
        raise ValueError(
            "Missing required anchor feature columns in input CSV:\n"
            + "\n".join(f"- {c}" for c in missing)
        )

    # Target
    y = df[args.target].astype(float)

    # Features (anchors only)
    X = df[ANCHOR_FEATURES].copy()

    # Safety: coerce to numeric (in case CSV parsing produces object dtype)
    for c in ANCHOR_FEATURES:
        X[c] = pd.to_numeric(X[c], errors="coerce")

    num_cols = ANCHOR_FEATURES  # explicit order = stable coef mapping

    pre = ColumnTransformer(
        transformers=[
            ("num", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ]), num_cols)
        ],
        remainder="drop"
    )

    pipe = Pipeline([
        ("pre", pre),
        ("model", Ridge(alpha=args.alpha, random_state=0))
    ])

    # Cross-val metrics
    kf = KFold(n_splits=min(args.folds, len(df)), shuffle=True, random_state=0)

    mae_scores = -cross_val_score(pipe, X, y, cv=kf, scoring="neg_mean_absolute_error")
    rmse_scores = np.sqrt(-cross_val_score(pipe, X, y, cv=kf, scoring="neg_mean_squared_error"))
    r2_scores = cross_val_score(pipe, X, y, cv=kf, scoring="r2")

    print("CV MAE  :", float(mae_scores.mean()), "+/-", float(mae_scores.std()))
    print("CV RMSE :", float(rmse_scores.mean()), "+/-", float(rmse_scores.std()))
    print("CV R2   :", float(r2_scores.mean()), "+/-", float(r2_scores.std()))

    # Fit on all data (for coefficient inspection)
    pipe.fit(X, y)
    yhat = pipe.predict(X)

    # Train-fit metrics (sanity only)
    print("\nTrain-fit MAE :", mean_absolute_error(y, yhat))
    print("Train-fit RMSE:", np.sqrt(mean_squared_error(y, yhat)))
    print("Train-fit R2  :", r2_score(y, yhat))

    # Extract coefficients (these map 1:1 to ANCHOR_FEATURES due to explicit ordering)
    coefs = pipe.named_steps["model"].coef_
    if len(coefs) != len(num_cols):
        raise RuntimeError(
            f"Coefficient length mismatch: got {len(coefs)} coefs but expected {len(num_cols)}."
        )

    coef_df = pd.DataFrame({
        "feature": num_cols,
        "coef": coefs
    }).sort_values("coef", ascending=False)

    coef_df.to_csv(args.out_coef, index=False)

    # Predictions output
    keep_cols = [c for c in ["video_id", "title", args.target] if c in df.columns]
    pred_df = df[keep_cols].copy()
    pred_df["snr_pred_trainfit"] = yhat
    pred_df.to_csv(args.out_pred, index=False)

    print(f"\nWrote coeffs: {args.out_coef}")
    print(f"Wrote preds : {args.out_pred}")


if __name__ == "__main__":
    main()