#!/usr/bin/env python3
import argparse
import numpy as np
import pandas as pd

from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine_csv", required=True, help="Path to reports/snr_engine_scores.csv")
    ap.add_argument("--out_pred", default="reports/calibrated_engine_preds.csv")
    ap.add_argument("--out_coef", default="reports/calibrated_engine_weights.csv")
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()

    df = pd.read_csv(args.engine_csv)

    req = ["snr_score", "clarity_z", "depth_z", "penalty_z"]
    missing = [c for c in req if c not in df.columns]
    if missing:
        raise ValueError("Missing required columns:\n" + "\n".join(f"- {c}" for c in missing))

    y = df["snr_score"].astype(float)
    X = df[["clarity_z", "depth_z", "penalty_z"]].copy()

    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", Ridge(alpha=args.alpha, random_state=0)),
    ])

    kf = KFold(n_splits=min(args.folds, len(df)), shuffle=True, random_state=0)
    mae = -cross_val_score(pipe, X, y, cv=kf, scoring="neg_mean_absolute_error")
    rmse = np.sqrt(-cross_val_score(pipe, X, y, cv=kf, scoring="neg_mean_squared_error"))
    r2 = cross_val_score(pipe, X, y, cv=kf, scoring="r2")

    print("CV MAE  :", float(mae.mean()), "+/-", float(mae.std()))
    print("CV RMSE :", float(rmse.mean()), "+/-", float(rmse.std()))
    print("CV R2   :", float(r2.mean()), "+/-", float(r2.std()))

    pipe.fit(X, y)
    preds = pipe.predict(X)

    out = df[["video_id", "title", "snr_score", "snr_engine_score", "clarity_z", "depth_z", "penalty_z"]].copy()
    out["snr_engine_calibrated_pred_trainfit"] = preds
    out.to_csv(args.out_pred, index=False)

    # Extract weights (in scaled space; still interpretable sign-wise)
    w = pipe.named_steps["model"].coef_
    b = pipe.named_steps["model"].intercept_

    coef_df = pd.DataFrame({
        "feature": ["clarity_z", "depth_z", "penalty_z"],
        "weight": w
    })
    coef_df.loc[len(coef_df)] = ["intercept", b]
    coef_df.to_csv(args.out_coef, index=False)

    print(f"\nWrote preds : {args.out_pred}")
    print(f"Wrote coefs : {args.out_coef}")

if __name__ == "__main__":
    main()
