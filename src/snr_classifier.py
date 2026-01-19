from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge

LABEL_COL = "snr_score"

FEATURE_PHRASES = {
    "recycled_penalty": "Residual generic phrasing (after explainer discount)",
    "recycled_signal_similarity": "Similarity to recycled content patterns",
    "structure_hits": "Clear structural organization",
    "content_density_proxy": "High information density",
    "self_promo": "Self-promotional framing",
    "modal_count": "Speculative / modal language usage",
    "you_count": "Direct address to viewer",
    "imperative_count": "Command-style language",
}

@dataclass
class SNRModel:
    pipe: Pipeline

EXCLUDE_COLS = {
    "snr_score",
    "takeaway_clarity_1_5",
    "insight_depth_1_5",
    "signal_level",
    "noise_superclass",
    "noise_subtype",
    "primary_topic",
}

def _numeric_cols(df: pd.DataFrame) -> list[str]:
    cols = []
    for c in df.columns:
        if c in ["video_id", "title", "url", "transcript"]:
            continue
        if c in EXCLUDE_COLS:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)
    return cols

def build_pipeline(X: pd.DataFrame, alpha: float = 1.0) -> Pipeline:
    num_cols = _numeric_cols(X)

    pre = ColumnTransformer(
        transformers=[
            ("num", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]), num_cols)
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    model = Ridge(alpha=alpha, random_state=42)
    return Pipeline([("pre", pre), ("model", model)])

def train_and_save(train_csv: Path, model_path: Path, alpha: float = 1.0) -> SNRModel:
    df = pd.read_csv(train_csv)
    if LABEL_COL not in df.columns:
        raise ValueError(f"Training CSV missing label column: {LABEL_COL}")

    y = df[LABEL_COL].astype(float).values
    X = df.drop(columns=[LABEL_COL])

    pipe = build_pipeline(X, alpha=alpha)
    pipe.fit(X, y)

    model = SNRModel(pipe=pipe)
    joblib.dump(model, model_path)
    return model

def load_or_train_model(train_csv: Path, model_path: Path, alpha: float = 1.0) -> SNRModel:
    if model_path.exists():
        return joblib.load(model_path)
    return train_and_save(train_csv, model_path, alpha=alpha)

def predict_score(model: SNRModel, X_eval: pd.DataFrame) -> float:
    yhat = model.pipe.predict(X_eval)
    return float(yhat[0])

def _pretty_feature_name(name: str) -> str:
    base = name.split("__")[-1]
    phrase = FEATURE_PHRASES.get(base)
    return f"{phrase} ({base})" if phrase else base

def explain_prediction(model: SNRModel, X_eval: pd.DataFrame, top_k_pos: int = 3, top_k_neg: int = 2):
    """
    Ridge explanation using contribution = coef * standardized_feature_value.
    Works because we’re using StandardScaler in the numeric pipeline.
    """
    pipe = model.pipe
    pre = pipe.named_steps["pre"]
    reg = pipe.named_steps["model"]

    X_trans = pre.transform(X_eval)  # (1, n_features)
    if hasattr(pre, "get_feature_names_out"):
        names = list(pre.get_feature_names_out())
    else:
        # fallback if sklearn version differs
        names = [f"f{i}" for i in range(X_trans.shape[1])]

    coef = reg.coef_
    contrib = (X_trans.toarray() if hasattr(X_trans, "toarray") else X_trans)[0] * coef

    idx_sorted = np.argsort(contrib)
    neg_idx = idx_sorted[:top_k_neg]
    pos_idx = idx_sorted[::-1][:top_k_pos]

    reasons = []
    for i in pos_idx:
        reasons.append(f"+ {_pretty_feature_name(names[i])}: +{contrib[i]:.3f}")
    for i in neg_idx:
        reasons.append(f"- {_pretty_feature_name(names[i])}: {contrib[i]:.3f}")
    return reasons