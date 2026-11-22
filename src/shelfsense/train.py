"""Train and evaluate ShelfSense models.

Usage:
    python -m shelfsense.train --raw data/raw/listings.csv
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (accuracy_score, f1_score, mean_absolute_error,
                             mean_absolute_percentage_error, r2_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .cleaning import load_clean

SEED = 42
TIERS = ["Low", "Medium", "High"]

NUM_DEMAND = ["price", "rating", "log_reviews", "discount_pct", "is_sponsored", "is_best_seller",
              "is_amazons_choice", "is_limited_deal", "has_coupon", "has_sustainability_badge",
              "is_renewed", "title_words"]
NUM_PRICE = ["rating", "log_reviews", "is_renewed", "title_words", "title_len", "is_sponsored"]


def demand_tier(units: pd.Series) -> pd.Series:
    """Low: <=100 sold/month, Medium: 200-999, High: >=1000."""
    return pd.cut(units, [-1, 100, 999, np.inf], labels=TIERS).astype("string")


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["log_reviews"] = np.log1p(df["reviews"].fillna(0))
    df["rating"] = df["rating"].fillna(df["rating"].median())
    return df


def _prep(num, with_text=True):
    parts = [("num", StandardScaler(), num),
             ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=5), ["category"])]
    if with_text:
        parts.append(("txt", TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=6000,
                                             stop_words="english", sublinear_tf=True), "title"))
    return ColumnTransformer(parts, sparse_threshold=1.0)


def train_demand(df: pd.DataFrame):
    d = df.dropna(subset=["units_sold_month", "price"]).copy()
    d["tier"] = demand_tier(d["units_sold_month"])
    X, y = d[NUM_DEMAND + ["category", "title"]], d["tier"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
    candidates = {
        "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced"),
        "Random Forest": RandomForestClassifier(n_estimators=150, min_samples_leaf=3, n_jobs=-1,
                                                class_weight="balanced", random_state=SEED),
        "LightGBM": LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=31,
                                   class_weight="balanced", random_state=SEED, verbose=-1),
    }
    results, fitted = {}, {}
    for name, clf in candidates.items():
        pipe = Pipeline([("prep", _prep(NUM_DEMAND, with_text=True)), ("clf", clf)])
        pipe.fit(Xtr, ytr)
        pred = pipe.predict(Xte)
        results[name] = {"accuracy": round(accuracy_score(yte, pred), 4),
                         "macro_f1": round(f1_score(yte, pred, average="macro"), 4)}
        fitted[name] = pipe
    best = max(results, key=lambda k: results[k]["macro_f1"])
    baseline = round(float((yte == ytr.mode().iat[0]).mean()), 4)
    return fitted[best], best, results, baseline, len(d)


def train_price(df: pd.DataFrame):
    d = df.dropna(subset=["price"]).copy()
    d = d[d["price"] <= d["price"].quantile(0.995)]
    X, y = d[NUM_PRICE + ["category", "title"]], np.log1p(d["price"])
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=SEED)
    candidates = {
        "Ridge": Ridge(alpha=1.0),
        "LightGBM": LGBMRegressor(n_estimators=600, learning_rate=0.05, num_leaves=63,
                                  random_state=SEED, verbose=-1),
    }
    results, fitted = {}, {}
    for name, reg in candidates.items():
        pipe = Pipeline([("prep", _prep(NUM_PRICE)), ("reg", reg)])
        pipe.fit(Xtr, ytr)
        pred = np.expm1(pipe.predict(Xte))
        true = np.expm1(yte)
        results[name] = {"mae_usd": round(mean_absolute_error(true, pred), 2),
                         "mape_pct": round(mean_absolute_percentage_error(true, pred) * 100, 1),
                         "r2_log_price": round(r2_score(yte, np.log1p(pred)), 4)}
        fitted[name] = pipe
    best = min(results, key=lambda k: results[k]["mae_usd"])
    base_mae = round(mean_absolute_error(np.expm1(yte), np.full(len(yte), np.expm1(ytr.median()))), 2)
    return fitted[best], best, results, base_mae, len(d)


def main(raw: str, out: str = "models", reports: str = "reports") -> dict:
    out_p, rep_p = Path(out), Path(reports)
    out_p.mkdir(exist_ok=True)
    rep_p.mkdir(exist_ok=True)
    df = add_features(load_clean(raw))

    d_model, d_best, d_res, d_base, d_n = train_demand(df)
    p_model, p_best, p_res, p_base, p_n = train_price(df)
    joblib.dump(d_model, out_p / "demand_model.joblib", compress=3)
    joblib.dump(p_model, out_p / "price_model.joblib", compress=3)

    metrics = {
        "rows_after_cleaning": int(len(df)),
        "demand": {"task": "3-class demand tier", "rows": d_n, "best_model": d_best,
                   "majority_baseline_accuracy": d_base, "results": d_res},
        "price": {"task": "price regression", "rows": p_n, "best_model": p_best,
                  "median_baseline_mae_usd": p_base, "results": p_res},
    }
    (rep_p / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))
    return metrics


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True, help="path to the raw listings CSV")
    ap.add_argument("--out", default="models")
    ap.add_argument("--reports", default="reports")
    a = ap.parse_args()
    main(a.raw, a.out, a.reports)
