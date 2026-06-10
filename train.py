"""
Train and compare ML models for stock direction prediction.

Usage:
    python train.py
    python train.py --tickers AAPL MSFT GOOGL --period 5y
"""
import argparse
import json
import logging
import sys

from src.pipelines.training_pipeline import TrainingPipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(description="Train stock direction ML models")
    parser.add_argument(
        "--tickers",
        nargs="+",
        default=None,
        help="Ticker symbols for training (default: config list)",
    )
    parser.add_argument(
        "--period",
        default="5y",
        help="yfinance history period (default: 5y)",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Model names to compare (default: all candidates)",
    )
    args = parser.parse_args()

    try:
        pipeline = TrainingPipeline()
        result = pipeline.run(
            tickers=args.tickers,
            period=args.period,
            model_names=args.models,
        )

        meta = result["metadata"]
        ds = meta["dataset_size"]

        print("\n=== Training Complete ===")
        print(f"Dataset: {ds['total_rows']} rows from {ds['tickers']} tickers")
        print(f"  train={ds['train_rows']}  val={ds['val_rows']}  test={ds['test_rows']}")
        print(f"Features: {meta['feature_count']} time-series features")
        print(f"Best model: {result['best_model']}")
        pm = result["primary_metric"]
        print(f"Validation {pm}: {result['best_val_metrics'][pm]:.4f}")
        print(f"Test {pm}: {result['best_test_metrics'][pm]:.4f}")
        print(f"Test accuracy: {result['best_test_metrics']['accuracy']:.4f}")

        print("\nModel comparison (test set):")
        for name, metrics in result["model_comparison"].items():
            test = metrics["test"]
            print(
                f"  {name:25s}  acc={test['accuracy']:.4f}  "
                f"f1={test['f1']:.4f}  auc={test['roc_auc']:.4f}"
            )

        print("\nBenchmark vs market references (test ROC-AUC):")
        for bench_name, comp in meta["benchmark_comparison"]["market_benchmarks"].items():
            delta = comp["delta"]["roc_auc"]
            sign = "+" if delta >= 0 else ""
            print(
                f"  vs {bench_name:30s}  ours={comp['our_model']['roc_auc']:.4f}  "
                f"baseline={comp['baseline']['roc_auc']:.4f}  ({sign}{delta:.4f})"
            )

        print("\nArtifacts saved to artifacts/models/")
        print("View MLflow UI: mlflow ui --backend-store-uri mlruns")

        return 0

    except Exception as exc:
        logger.exception("Training failed: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
