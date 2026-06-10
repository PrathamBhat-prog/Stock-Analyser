import json
import logging
import os
from dataclasses import dataclass

import joblib
import mlflow
import pandas as pd
from sklearn.ensemble import (
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
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

logger = logging.getLogger(__name__)


@dataclass
class ModelResult:
    name: str
    model_type: str
    metrics: dict
    sklearn_pipeline: Pipeline | None = None
    lstm_trainer: LSTMTrainer | None = None


def _build_sklearn_pipelines() -> dict[str, Pipeline]:
    return {
        "logistic_regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)),
        ]),
        "random_forest": Pipeline([
            ("model", RandomForestClassifier(
                n_estimators=200,
                max_depth=8,
                class_weight="balanced_subsample",
                random_state=42,
                n_jobs=-1,
            )),
        ]),
        "gradient_boosting": Pipeline([
            ("model", GradientBoostingClassifier(random_state=42)),
        ]),
        "hist_gradient_boosting": Pipeline([
            ("model", HistGradientBoostingClassifier(
                max_depth=6,
                learning_rate=0.05,
                max_iter=200,
                random_state=42,
            )),
        ]),
    }


def _evaluate_sklearn(model: Pipeline, x: pd.DataFrame, y: pd.Series) -> dict:
    preds = model.predict(x)
    probas = model.predict_proba(x)[:, 1]
    return _metrics_from_arrays(y, preds, probas)


def _metrics_from_arrays(y_true, preds, probas) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, preds)),
        "precision": float(precision_score(y_true, preds, zero_division=0)),
        "recall": float(recall_score(y_true, preds, zero_division=0)),
        "f1": float(f1_score(y_true, preds, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probas)),
    }


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


def train_and_compare(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    model_names: list[str] | None = None,
    primary_metric: str = PRIMARY_METRIC,
) -> tuple[ModelResult, list[ModelResult]]:
    x_train = train_df[FEATURE_COLUMNS]
    y_train = train_df[TARGET_COLUMN]
    x_val = val_df[FEATURE_COLUMNS]
    y_val = val_df[TARGET_COLUMN]
    x_test = test_df[FEATURE_COLUMNS]
    y_test = test_df[TARGET_COLUMN]

    sklearn_models = _build_sklearn_pipelines()
    if model_names:
        sklearn_models = {k: v for k, v in sklearn_models.items() if k in model_names}

    results: list[ModelResult] = []

    for name, pipeline in sklearn_models.items():
        logger.info("Training %s ...", name)
        pipeline.fit(x_train, y_train)

        val_metrics = _evaluate_sklearn(pipeline, x_val, y_val)
        test_metrics = _evaluate_sklearn(pipeline, x_test, y_test)

        with mlflow.start_run(run_name=name, nested=True):
            mlflow.log_param("model_name", name)
            mlflow.log_param("model_type", "sklearn")
            for k, v in val_metrics.items():
                mlflow.log_metric(f"val_{k}", v)
            for k, v in test_metrics.items():
                mlflow.log_metric(f"test_{k}", v)
            mlflow.sklearn.log_model(pipeline, artifact_path="model")

        results.append(ModelResult(
            name=name,
            model_type="sklearn",
            sklearn_pipeline=pipeline,
            metrics={"val": val_metrics, "test": test_metrics},
        ))
        logger.info(
            "%s — val %s: %.4f, test %s: %.4f",
            name, primary_metric, val_metrics[primary_metric],
            primary_metric, test_metrics[primary_metric],
        )

    if model_names is None or "lstm" in model_names:
        logger.info("Training lstm ...")
        lstm = LSTMTrainer()
        lstm.fit(train_df, val_df)
        val_metrics = lstm.evaluate_df(val_df)
        test_metrics = lstm.evaluate_df(test_df)

        with mlflow.start_run(run_name="lstm", nested=True):
            mlflow.log_param("model_name", "lstm")
            mlflow.log_param("model_type", "lstm")
            for k, v in val_metrics.items():
                mlflow.log_metric(f"val_{k}", v)
            for k, v in test_metrics.items():
                mlflow.log_metric(f"test_{k}", v)

        results.append(ModelResult(
            name="lstm",
            model_type="lstm",
            lstm_trainer=lstm,
            metrics={"val": val_metrics, "test": test_metrics},
        ))
        logger.info(
            "lstm — val %s: %.4f, test %s: %.4f",
            primary_metric, val_metrics[primary_metric],
            primary_metric, test_metrics[primary_metric],
        )

    best = max(results, key=lambda r: r.metrics["val"][primary_metric])
    logger.info(
        "Best model: %s (val %s=%.4f, test %s=%.4f)",
        best.name,
        primary_metric,
        best.metrics["val"][primary_metric],
        primary_metric,
        best.metrics["test"][primary_metric],
    )
    return best, results


def _build_benchmark_report(best: ModelResult) -> dict:
    test = best.metrics["test"]
    comparisons = {}
    for name, baseline in MARKET_BENCHMARKS.items():
        comparisons[name] = {
            "baseline": baseline,
            "our_model": test,
            "delta": {m: round(test[m] - baseline[m], 4) for m in baseline},
        }
    return {
        "our_best_model": best.name,
        "our_test_metrics": test,
        "market_benchmarks": comparisons,
        "notes": (
            "Baselines are directional-classification references from literature. "
            "Primary selection metric is ROC-AUC to reduce class-imbalance bias."
        ),
    }


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
        "best_model_name": best.name,
        "model_type": best.model_type,
        "model_path": model_path,
        "primary_metric": PRIMARY_METRIC,
        "val_metrics": best.metrics["val"],
        "test_metrics": best.metrics["test"],
        "feature_columns": FEATURE_COLUMNS,
        "feature_count": len(FEATURE_COLUMNS),
        "target_column": TARGET_COLUMN,
        "forecast_horizon_days": forecast_horizon,
        "dataset_size": {
            "train_rows": train_rows,
            "val_rows": val_rows,
            "test_rows": test_rows,
            "total_rows": train_rows + val_rows + test_rows,
            "tickers": len(tickers),
        },
        "tickers": tickers,
        "benchmark_comparison": benchmark_report,
    }

    with open(MODEL_METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    with open(BENCHMARK_REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(benchmark_report, f, indent=2)

    mlflow.log_param("best_model", best.name)
    mlflow.log_param("best_model_type", best.model_type)
    for k, v in best.metrics["val"].items():
        mlflow.log_metric(f"best_val_{k}", v)
    for k, v in best.metrics["test"].items():
        mlflow.log_metric(f"best_test_{k}", v)

    mlflow.log_artifact(MODEL_METADATA_PATH)
    mlflow.log_artifact(BENCHMARK_REPORT_PATH)
    if best.model_type == "lstm":
        mlflow.log_artifact(BEST_LSTM_PATH)
        mlflow.log_artifact(LSTM_SCALER_PATH)
    elif os.path.exists(FEATURE_IMPORTANCE_PATH):
        mlflow.log_artifact(FEATURE_IMPORTANCE_PATH)

    return metadata
