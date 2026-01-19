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

    # Keep only numeric feature columns + target
    y = df[args.target].astype(float)

    # Drop label columns / text columns from features
    drop_cols = {
        "snr_score","takeaway_clarity_1_5","insight_depth_1_5",
        "signal_level","noise_superclass","noise_subtype","primary_topic",
        "short_reasoning","notes","title","video_id"
    }
    X = df[[c for c in df.columns if c not in drop_cols]].copy()

    # Use numeric columns only
    num_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    X = X[num_cols]

    pre = ColumnTransformer(
        transformers=[
            ("num", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ]), num_cols)
        ],
        remainder="drop"
    )

    model = Ridge(alpha=args.alpha, random_state=0)

    pipe = Pipeline([
        ("pre", pre),
        ("model", model)
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

    # Train-fit metrics (not for bragging; just for sanity)
    print("\nTrain-fit MAE :", mean_absolute_error(y, yhat))
    print("Train-fit RMSE:", np.sqrt(mean_squared_error(y, yhat)))
    print("Train-fit R2  :", r2_score(y, yhat))

    # Extract coefficients (mapped to num_cols)
    coefs = pipe.named_steps["model"].coef_
    coef_df = pd.DataFrame({
        "feature": num_cols,
        "coef": coefs
    }).sort_values("coef", ascending=False)

    coef_df.to_csv(args.out_coef, index=False)

    pred_df = df[["video_id","title","snr_score"]].copy()
    pred_df["snr_pred_trainfit"] = yhat
    pred_df.to_csv(args.out_pred, index=False)

    print(f"\nWrote coeffs: {args.out_coef}")
    print(f"Wrote preds : {args.out_pred}")

if __name__ == "__main__":
    main()
