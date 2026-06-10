"""
PyTorch LSTM classifier for stock direction prediction.

Architecture:
    StockLSTM:
      - LSTM(input_size, hidden=128, layers=2, dropout=0.3)
      - Dropout(0.3)
      - Linear(128, 1)   → BCEWithLogitsLoss

Training features:
  - Class-imbalance correction via pos_weight in BCEWithLogitsLoss
  - Early stopping (patience configurable via LSTM_PATIENCE)
  - Epoch-by-epoch loss + val-ROC-AUC tracking
  - Generates epoch_plot.png (train loss & val ROC-AUC curves) at
    docs/training_results/ for proof of training
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")   # non-interactive backend — safe on headless servers
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from src.config.ml_config import (
    BEST_LSTM_PATH,
    LSTM_BATCH_SIZE,
    LSTM_DROPOUT,
    LSTM_EPOCHS,
    LSTM_HIDDEN_SIZE,
    LSTM_LEARNING_RATE,
    LSTM_NUM_LAYERS,
    LSTM_PATIENCE,
    LSTM_SCALER_PATH,
    SEQUENCE_LENGTH,
)
from src.data.sequences import build_sequences
from src.models.feature_columns import FEATURE_COLUMNS, TARGET_COLUMN
from src.models.metrics import compute_all

logger = logging.getLogger(__name__)

# Output directory for epoch plots
_PLOT_DIR = Path("docs") / "training_results"


# ── Model definition ──────────────────────────────────────────────────────────

class StockLSTM(nn.Module):
    """
    Bidirectional-optional LSTM for binary stock direction classification.

    Input  : (batch, seq_len, n_features)
    Output : (batch, 1) — raw logit (apply sigmoid for probability)
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.layer_norm = nn.LayerNorm(hidden_size)   # stabilise LSTM output
        self.dropout    = nn.Dropout(dropout)
        self.fc         = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        out    = self.layer_norm(out[:, -1, :])   # take last timestep
        out    = self.dropout(out)
        return self.fc(out)


# ── Trainer ───────────────────────────────────────────────────────────────────

class LSTMTrainer:
    """Train, evaluate, save, and load an LSTM direction classifier."""

    def __init__(
        self,
        seq_len:     int   = SEQUENCE_LENGTH,
        hidden_size: int   = LSTM_HIDDEN_SIZE,
        num_layers:  int   = LSTM_NUM_LAYERS,
        dropout:     float = LSTM_DROPOUT,
    ):
        self.seq_len     = seq_len
        self.hidden_size = hidden_size
        self.num_layers  = num_layers
        self.dropout     = dropout
        self.scaler      = StandardScaler()
        self.model: StockLSTM | None = None
        self.device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("LSTM device: %s", self.device)

    # ── Data helpers ──────────────────────────────────────────────────────────

    def _scale_sequences(self, x: np.ndarray, fit: bool = False) -> np.ndarray:
        n, t, f = x.shape
        flat    = x.reshape(-1, f)
        scaled  = self.scaler.fit_transform(flat) if fit else self.scaler.transform(flat)
        return scaled.reshape(n, t, f).astype(np.float32)

    def _to_loader(self, x: np.ndarray, y: np.ndarray, shuffle: bool) -> DataLoader:
        dataset = TensorDataset(
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32),
        )
        return DataLoader(dataset, batch_size=LSTM_BATCH_SIZE, shuffle=shuffle, pin_memory=False)

    # ── Epoch plot ────────────────────────────────────────────────────────────

    @staticmethod
    def _save_epoch_plot(
        train_losses:   list[float],
        val_aucs:       list[float],
        stopped_epoch:  int,
    ) -> None:
        """
        Generate a dual-axis epoch plot:
          - Left axis  : train BCE loss (blue)
          - Right axis : validation ROC-AUC (orange)
        Vertical dashed line marks the best (restored) epoch.
        Saved to docs/training_results/epoch_plot.png.
        """
        _PLOT_DIR.mkdir(parents=True, exist_ok=True)
        save_path = _PLOT_DIR / "epoch_plot.png"

        epochs = list(range(1, len(train_losses) + 1))
        best_epoch = int(np.argmax(val_aucs)) + 1

        fig, ax1 = plt.subplots(figsize=(12, 6))
        ax2 = ax1.twinx()

        # Train loss
        ax1.plot(epochs, train_losses, color="#3B82F6", linewidth=2, label="Train Loss (BCE)")
        ax1.set_xlabel("Epoch", fontsize=13)
        ax1.set_ylabel("Training Loss (BCE)", color="#3B82F6", fontsize=12)
        ax1.tick_params(axis="y", labelcolor="#3B82F6")

        # Val AUC
        ax2.plot(epochs, val_aucs, color="#F97316", linewidth=2, label="Val ROC-AUC")
        ax2.set_ylabel("Validation ROC-AUC", color="#F97316", fontsize=12)
        ax2.tick_params(axis="y", labelcolor="#F97316")
        ax2.set_ylim(0.45, 1.0)

        # Best epoch marker
        ax1.axvline(x=best_epoch, color="#10B981", linestyle="--", linewidth=1.5,
                    label=f"Best epoch ({best_epoch})")
        if stopped_epoch < len(epochs):
            ax1.axvline(x=stopped_epoch, color="#EF4444", linestyle=":", linewidth=1.5,
                        label=f"Early stop ({stopped_epoch})")

        # Annotation: best AUC
        best_auc = val_aucs[best_epoch - 1]
        ax2.annotate(
            f"Best AUC\n{best_auc:.4f}",
            xy=(best_epoch, best_auc),
            xytext=(best_epoch + max(1, len(epochs) * 0.05), best_auc - 0.015),
            fontsize=10,
            color="#F97316",
            arrowprops=dict(arrowstyle="->", color="#F97316"),
        )

        # Legend (combine both axes)
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower right", fontsize=10)

        plt.title(
            f"LSTM Training — {len(epochs)} epochs  |  10y OHLCV data  |  "
            f"Best val ROC-AUC: {best_auc:.4f}",
            fontsize=14,
            fontweight="bold",
        )
        plt.tight_layout()
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
        logger.info("Epoch plot saved → %s", save_path)

    # ── Training ──────────────────────────────────────────────────────────────

    def fit(self, train_df, val_df) -> dict:
        x_train, y_train = build_sequences(train_df, self.seq_len)
        x_val,   y_val   = build_sequences(val_df,   self.seq_len)

        x_train = self._scale_sequences(x_train, fit=True)
        x_val   = self._scale_sequences(x_val,   fit=False)

        train_loader = self._to_loader(x_train, y_train, shuffle=True)
        val_loader   = self._to_loader(x_val,   y_val,   shuffle=False)

        input_size = len(FEATURE_COLUMNS)
        self.model = StockLSTM(
            input_size=input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
        ).to(self.device)

        # Class-imbalance correction
        pos        = float(y_train.sum())
        neg        = float(len(y_train) - pos)
        pos_weight = torch.tensor([neg / max(pos, 1)], device=self.device)
        criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer  = torch.optim.Adam(self.model.parameters(), lr=LSTM_LEARNING_RATE)
        scheduler  = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=3, verbose=False
        )

        best_val_auc     = -1.0
        patience_counter = 0
        best_state       = None
        stopped_epoch    = LSTM_EPOCHS

        train_losses: list[float] = []
        val_aucs:     list[float] = []

        for epoch in range(1, LSTM_EPOCHS + 1):
            # ── Training step ─────────────────────────────────────────────────
            self.model.train()
            epoch_loss = 0.0
            n_batches  = 0
            for xb, yb in train_loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                optimizer.zero_grad()
                logits = self.model(xb).squeeze(-1)
                loss   = criterion(logits, yb)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                epoch_loss += loss.item()
                n_batches  += 1

            avg_loss = epoch_loss / max(n_batches, 1)
            train_losses.append(avg_loss)

            # ── Validation step ───────────────────────────────────────────────
            val_metrics = self.evaluate_sequences(x_val, y_val)
            val_auc     = val_metrics["roc_auc"]
            val_aucs.append(val_auc)

            scheduler.step(val_auc)

            logger.info(
                "LSTM epoch %3d/%d — loss: %.4f  val_auc: %.4f  val_acc: %.4f  sharpe: %.3f",
                epoch, LSTM_EPOCHS,
                avg_loss,
                val_auc,
                val_metrics["accuracy"],
                val_metrics.get("sharpe_ratio", float("nan")),
            )

            # ── Early stopping ────────────────────────────────────────────────
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_state   = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= LSTM_PATIENCE:
                    stopped_epoch = epoch
                    logger.info("Early stopping at epoch %d (patience=%d)", epoch, LSTM_PATIENCE)
                    break

        if best_state:
            self.model.load_state_dict(best_state)

        # ── Save epoch plot as proof of training ──────────────────────────────
        self._save_epoch_plot(train_losses, val_aucs, stopped_epoch)

        return {"best_val_roc_auc": best_val_auc, "epochs_run": len(train_losses)}

    # ── Evaluation ────────────────────────────────────────────────────────────

    def evaluate_sequences(self, x: np.ndarray, y: np.ndarray) -> dict:
        if self.model is None:
            raise RuntimeError("Model not trained.")

        self.model.eval()
        loader = self._to_loader(x, y, shuffle=False)
        probas: list[float] = []

        with torch.no_grad():
            for xb, _ in loader:
                xb        = xb.to(self.device)
                logits    = self.model(xb).squeeze(-1)
                batch_p   = torch.sigmoid(logits).cpu().numpy()
                probas.extend(batch_p.tolist())

        probas_arr = np.array(probas)
        preds      = (probas_arr >= 0.5).astype(int)
        return compute_all(y, preds, probas_arr)

    def evaluate_df(self, df) -> dict:
        x, y = build_sequences(df, self.seq_len)
        x    = self._scale_sequences(x, fit=False)
        return self.evaluate_sequences(x, y)

    def predict_proba_latest(self, df) -> float:
        if self.model is None:
            raise RuntimeError("Model not trained or loaded.")

        feature_df = df[FEATURE_COLUMNS].dropna()
        if len(feature_df) < self.seq_len:
            raise ValueError(
                f"Need at least {self.seq_len} rows of feature history for LSTM inference."
            )

        seq = feature_df.iloc[-self.seq_len:][FEATURE_COLUMNS].values.astype(np.float32)
        seq = self.scaler.transform(seq).reshape(1, self.seq_len, -1)

        self.model.eval()
        with torch.no_grad():
            x     = torch.tensor(seq, dtype=torch.float32).to(self.device)
            logit = self.model(x).squeeze(-1)
            return float(torch.sigmoid(logit).cpu().item())

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(
        self,
        model_path:  str = BEST_LSTM_PATH,
        scaler_path: str = LSTM_SCALER_PATH,
    ) -> None:
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        torch.save({
            "state_dict":  self.model.state_dict(),
            "seq_len":     self.seq_len,
            "hidden_size": self.hidden_size,
            "num_layers":  self.num_layers,
            "dropout":     self.dropout,
            "input_size":  len(FEATURE_COLUMNS),
        }, model_path)
        joblib.dump(self.scaler, scaler_path)

    @classmethod
    def load(
        cls,
        model_path:  str = BEST_LSTM_PATH,
        scaler_path: str = LSTM_SCALER_PATH,
    ) -> "LSTMTrainer":
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
        trainer    = cls(
            seq_len     = checkpoint["seq_len"],
            hidden_size = checkpoint["hidden_size"],
            num_layers  = checkpoint["num_layers"],
            dropout     = checkpoint["dropout"],
        )
        trainer.scaler = joblib.load(scaler_path)
        trainer.model  = StockLSTM(
            input_size  = checkpoint["input_size"],
            hidden_size = checkpoint["hidden_size"],
            num_layers  = checkpoint["num_layers"],
            dropout     = checkpoint["dropout"],
        )
        trainer.model.load_state_dict(checkpoint["state_dict"])
        trainer.model.to(trainer.device)
        trainer.model.eval()
        return trainer
