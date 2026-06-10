import logging
import os

import mlflow

from src.config.ml_config import (
    DEFAULT_TRAIN_TICKERS,
    FORECAST_HORIZON_DAYS,
    MLFLOW_EXPERIMENT_TRAINING,
    MODEL_CANDIDATES,
    PRIMARY_METRIC,
    TRAIN_PERIOD,
    TRAIN_RATIO,
    VAL_RATIO,
)
from src.data.dataset import build_training_dataset, chronological_split
from src.models.trainer import save_best_model, train_and_compare

logger = logging.getLogger(__name__)


def _setup_mlflow() -> str:
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    mlruns_path = os.path.join(project_root, "mlruns")
    mlflow.set_tracking_uri(f"file:///{mlruns_path}")
    mlflow.set_experiment(MLFLOW_EXPERIMENT_TRAINING)
    return project_root


class TrainingPipeline:
    """
    Full ML engineering workflow:
    1. Fetch multi-ticker data (yfinance)
    2. Feature engineering + labeling
    3. Chronological train/val/test split
    4. Train and compare multiple models
    5. Select best by validation metric
    6. Log everything to MLflow and save artifacts
    """

    def run(
        self,
        tickers: list[str] | None = None,
        period: str = TRAIN_PERIOD,
        model_names: list[str] | None = None,
    ) -> dict:
        tickers = tickers or DEFAULT_TRAIN_TICKERS
        model_names = model_names or MODEL_CANDIDATES

        _setup_mlflow()

        logger.info("Building dataset from %d tickers ...", len(tickers))
        dataset = build_training_dataset(tickers=tickers, period=period)
        train_df, val_df, test_df = chronological_split(
            dataset, train_ratio=TRAIN_RATIO, val_ratio=VAL_RATIO
        )

        logger.info(
            "Dataset rows — train: %d, val: %d, test: %d",
            len(train_df), len(val_df), len(test_df),
        )

        with mlflow.start_run(run_name="model-comparison"):
            mlflow.log_param("tickers", ",".join(tickers))
            mlflow.log_param("period", period)
            mlflow.log_param("forecast_horizon_days", FORECAST_HORIZON_DAYS)
            mlflow.log_param("primary_metric", PRIMARY_METRIC)
            mlflow.log_param("models", ",".join(model_names))
            mlflow.log_metric("total_rows", len(dataset))
            mlflow.log_metric("train_rows", len(train_df))
            mlflow.log_metric("val_rows", len(val_df))
            mlflow.log_metric("test_rows", len(test_df))

            best, all_results = train_and_compare(
                train_df=train_df,
                val_df=val_df,
                test_df=test_df,
                model_names=model_names,
                primary_metric=PRIMARY_METRIC,
            )

            metadata = save_best_model(
                best=best,
                train_rows=len(train_df),
                val_rows=len(val_df),
                test_rows=len(test_df),
                tickers=tickers,
                forecast_horizon=FORECAST_HORIZON_DAYS,
            )

        comparison = {
            name: {
                "val": res.metrics["val"],
                "test": res.metrics["test"],
            }
            for name, res in ((r.name, r) for r in all_results)
        }

        return {
            "best_model": best.name,
            "primary_metric": PRIMARY_METRIC,
            "best_val_metrics": best.metrics["val"],
            "best_test_metrics": best.metrics["test"],
            "model_comparison": comparison,
            "metadata": metadata,
        }
