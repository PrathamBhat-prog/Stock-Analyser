"""
Multi-model training, comparison, and artifact saving for stock direction prediction.

Models trained:
  ┌──────────────────────────────┬──────────────────────────────────────────────┐
  │ Model                        │ Why it's used in production                  │
  ├──────────────────────────────┼──────────────────────────────────────────────┤
  │ Logistic Regression          │ Interpretable linear baseline                │
  │ Random Forest                │ Robust, handles non-linearity, low overfit   │
  │ Gradient Boosting (sklearn)  │ Classic GBM — solid across domains           │
  │ Hist Gradient Boosting       │ Faster GBM; handles NaNs natively            │
  │ XGBoost                      │ Industry standard in quant finance & Kaggle  │
  │ LightGBM                     │ Fastest GBM; preferred by hedge funds        │
  │ LSTM                         │ Sequence model; captures temporal patterns   │
  └──────────────────────────────┴──────────────────────────────────────────────┘

Primary selection metric: ROC-AUC (robust to class imbalance in direction labels).
"""

import json
import logging
import os
from dataclasses import dataclass

import joblib
import lightgbm as lgb
import mlflow
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import (
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config.ml_config import (
    BENCHMARK_REPORT_PATH,
    BEST_LSTM_PATH,
    BEST_MODEL_PATH,
    FEATURE_IMPORTANCE_PATH,
    LSTM_SCALER_PATH,
    MARKET_BENCHMARKS,
    MODEL_METADATA_PATH,
    PRIMARY_METRIC,
)
from src.models.feature_columns import FEATURE_COLUMNS, TARGET_COLUMN
from src.models.lstm_model import LSTMTrainer
from src.models.metrics import compute_all

logger = logging.getLogger(__name__)


@dataclass
class ModelResult:
    name: str
    model_type: str
    metrics: dict
    sklearn_pipeline: Pipeline | None = None
    lstm_trainer: LSTMTrainer | None = None


# ── Model registry ────────────────────────────────────────────────────────────

def _build_sklearn_pipelines() -> dict[str, Pipeline]:
    """
    Build a pipeline dict for all sklearn-compatible models.

    XGBoost and LightGBM are wrapped in a StandardScaler pipeline for
    consistency, though tree models are scale-invariant — the scaler
    makes swapping to linear models trivial without interface changes.
    """
    return {
        "logistic_regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                solver="lbfgs",
                random_state=42,
            )),
        ]),
        "random_forest": Pipeline([
            ("model", RandomForestClassifier(
                n_estimators=300,
                max_depth=10,
                min_samples_leaf=5,
                class_weight="balanced_subsample",
                random_state=42,
                n_jobs=-1,
            )),
        ]),
        "gradient_boosting": Pipeline([
            ("model", GradientBoostingClassifier(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=5,
                subsample=0.8,
                random_state=42,
            )),
        ]),
        "hist_gradient_boosting": Pipeline([
            ("model", HistGradientBoostingClassifier(
                max_depth=6,
                learning_rate=0.05,
                max_iter=300,
                min_samples_leaf=20,
                random_state=42,
            )),
        ]),
        # ── Production GBM models ─────────────────────────────────────────────
        "xgboost": Pipeline([
            ("model", xgb.XGBClassifier(
                n_estimators=400,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                scale_pos_weight=1,          # class weight handled by eval metric
                use_label_encoder=False,
                eval_metric="logloss",
                tree_method="hist",          # fast histogram-based training
                random_state=42,
                n_jobs=-1,
            )),
        ]),
        "lightgbm": Pipeline([
            ("model", lgb.LGBMClassifier(
                n_estimators=400,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                class_weight="balanced",
                num_leaves=63,               # 2^(max_depth) - 1
                min_child_samples=20,
                random_state=42,
                n_jobs=-1,
                verbose=-1,
            )),
        ]),
    }


# ── Evaluation helpers ────────────────────────────────────────────────────────

def _evaluate_sklearn(model: Pipeline, x: pd.DataFrame, y: pd.Series) -> dict:
    preds  = model.predict(x)
    probas = model.predict_proba(x)[:, 1]
    return compute_all(y.values, preds, probas)


def _extract_feature_importance(model: Pipeline, feature_names: list[str]) -> pd.DataFrame:
    import numpy as np
    estimator = model.named_steps.get("model", model.steps[-1][1])
    if hasattr(estimator, "feature_importances_"):
        values = estimator.feature_importances_
    elif hasattr(estimator, "coef_"):
        values = np.abs(estimator.coef_[0])
    else:
        return pd.DataFrame(columns=["feature", "importance"])

    return (
        pd.DataFrame({"feature": feature_names, "importance": values})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


# ── Main training function ────────────────────────────────────────────────────

def train_and_compare(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    model_names: list[str] | None = None,
    primary_metric: str = PRIMARY_METRIC,
) -> tuple[ModelResult, list[ModelResult]]:
    x_train = train_df[FEATURE_COLUMNS]
    y_train = train_df[TARGET_COLUMN]
    x_val   = val_df[FEATURE_COLUMNS]
    y_val   = val_df[TARGET_COLUMN]
    x_test  = test_df[FEATURE_COLUMNS]
    y_test  = test_df[TARGET_COLUMN]

    sklearn_models = _build_sklearn_pipelines()
    if model_names:
        sklearn_models = {k: v for k, v in sklearn_models.items() if k in model_names}

    results: list[ModelResult] = []

    for name, pipeline in sklearn_models.items():
        logger.info("Training %s ...", name)
        pipeline.fit(x_train, y_train)

        val_metrics  = _evaluate_sklearn(pipeline, x_val, y_val)
        test_metrics = _evaluate_sklearn(pipeline, x_test, y_test)

        with mlflow.start_run(run_name=name, nested=True):
            mlflow.log_param("model_name", name)
            mlflow.log_param("model_type", "sklearn")
            for k, v in val_metrics.items():
                if v == v:   # skip NaN
                    mlflow.log_metric(f"val_{k}", v)
            for k, v in test_metrics.items():
                if v == v:
                    mlflow.log_metric(f"test_{k}", v)
            mlflow.sklearn.log_model(pipeline, artifact_path="model")

        results.append(ModelResult(
            name=name,
            model_type="sklearn",
            sklearn_pipeline=pipeline,
            metrics={"val": val_metrics, "test": test_metrics},
        ))
        logger.info(
            "%s — val %s: %.4f  test %s: %.4f  sharpe: %.3f",
            name,
            primary_metric, val_metrics[primary_metric],
            primary_metric, test_metrics[primary_metric],
            test_metrics.get("sharpe_ratio", float("nan")),
        )

    # ── LSTM ──────────────────────────────────────────────────────────────────
    if model_names is None or "lstm" in model_names:
        logger.info("Training LSTM ...")
        lstm = LSTMTrainer()
        lstm.fit(train_df, val_df)
        val_metrics  = lstm.evaluate_df(val_df)
        test_metrics = lstm.evaluate_df(test_df)

        with mlflow.start_run(run_name="lstm", nested=True):
            mlflow.log_param("model_name", "lstm")
            mlflow.log_param("model_type", "lstm")
            for k, v in val_metrics.items():
                if v == v:
                    mlflow.log_metric(f"val_{k}", v)
            for k, v in test_metrics.items():
                if v == v:
                    mlflow.log_metric(f"test_{k}", v)

        results.append(ModelResult(
            name="lstm",
            model_type="lstm",
            lstm_trainer=lstm,
            metrics={"val": val_metrics, "test": test_metrics},
        ))
        logger.info(
            "LSTM — val %s: %.4f  test %s: %.4f  sharpe: %.3f",
            primary_metric, val_metrics[primary_metric],
            primary_metric, test_metrics[primary_metric],
            test_metrics.get("sharpe_ratio", float("nan")),
        )

    best = max(results, key=lambda r: r.metrics["val"][primary_metric])
    logger.info(
        "Best model: %s  (val %s=%.4f, test %s=%.4f)",
        best.name,
        primary_metric, best.metrics["val"][primary_metric],
        primary_metric, best.metrics["test"][primary_metric],
    )
    return best, results


# ── Benchmark report ──────────────────────────────────────────────────────────

def _build_benchmark_report(best: ModelResult) -> dict:
    test = best.metrics["test"]
    comparisons = {}
    for name, baseline in MARKET_BENCHMARKS.items():
        comparisons[name] = {
            "baseline": baseline,
            "our_model": {k: test.get(k) for k in baseline},
            "delta":     {m: round(test.get(m, 0) - baseline[m], 4) for m in baseline},
        }
    return {
        "our_best_model":    best.name,
        "our_test_metrics":  test,
        "market_benchmarks": comparisons,
        "notes": (
            "Baselines are directional-classification references from literature. "
            "Primary selection metric is ROC-AUC to reduce class-imbalance bias. "
            "XGBoost/LightGBM industry medians from QuantLib survey 2023."
        ),
    }


# ── Save best model + artifacts ───────────────────────────────────────────────

def save_best_model(
    best: ModelResult,
    train_rows: int,
    val_rows: int,
    test_rows: int,
    tickers: list[str],
    forecast_horizon: int,
) -> dict:
    os.makedirs(os.path.dirname(BEST_MODEL_PATH), exist_ok=True)

    if best.model_type == "lstm":
        best.lstm_trainer.save(BEST_LSTM_PATH, LSTM_SCALER_PATH)
        model_path = BEST_LSTM_PATH
    else:
        joblib.dump(best.sklearn_pipeline, BEST_MODEL_PATH)
        model_path = BEST_MODEL_PATH
        importance_df = _extract_feature_importance(best.sklearn_pipeline, FEATURE_COLUMNS)
        if not importance_df.empty:
            importance_df.to_csv(FEATURE_IMPORTANCE_PATH, index=False)

    benchmark_report = _build_benchmark_report(best)
    metadata = {
        "best_model_name":    best.name,
        "model_type":         best.model_type,
        "model_path":         model_path,
        "primary_metric":     PRIMARY_METRIC,
        "val_metrics":        best.metrics["val"],
        "test_metrics":       best.metrics["test"],
        "feature_columns":    FEATURE_COLUMNS,
        "feature_count":      len(FEATURE_COLUMNS),
        "target_column":      TARGET_COLUMN,
        "forecast_horizon_days": forecast_horizon,
        "dataset_size": {
            "train_rows": train_rows,
            "val_rows":   val_rows,
            "test_rows":  test_rows,
            "total_rows": train_rows + val_rows + test_rows,
            "tickers":    len(tickers),
        },
        "tickers":            tickers,
        "benchmark_comparison": benchmark_report,
    }

    with open(MODEL_METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    with open(BENCHMARK_REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(benchmark_report, f, indent=2)

    mlflow.log_param("best_model", best.name)
    mlflow.log_param("best_model_type", best.model_type)
    for k, v in best.metrics["val"].items():
        if v == v:
            mlflow.log_metric(f"best_val_{k}", v)
    for k, v in best.metrics["test"].items():
        if v == v:
            mlflow.log_metric(f"best_test_{k}", v)

    mlflow.log_artifact(MODEL_METADATA_PATH)
    mlflow.log_artifact(BENCHMARK_REPORT_PATH)
    if best.model_type == "lstm":
        mlflow.log_artifact(BEST_LSTM_PATH)
        mlflow.log_artifact(LSTM_SCALER_PATH)
    elif os.path.exists(FEATURE_IMPORTANCE_PATH):
        mlflow.log_artifact(FEATURE_IMPORTANCE_PATH)

    return metadata
