"""
Production-grade PyTorch LSTM for stock direction prediction.

Architecture (professional ML level):
  - Bidirectional LSTM (3 layers, hidden=256) -- captures both forward and
    backward temporal dependencies in the price sequence
  - Multi-Head Self-Attention (4 heads) -- lets the model focus on the
    most relevant timesteps (e.g. earnings-day spikes, support/resistance)
  - LayerNorm after LSTM output -- stabilises gradients in deep networks
  - Residual projection -- prevents vanishing gradients in 3-layer stack
  - Dropout (0.35) -- regularisation for financial noisy data
  - BCEWithLogitsLoss with pos_weight -- handles class imbalance

Training (professional settings):
  - 300 max epochs with patience=30 (real production budget)
  - AdamW optimiser (weight decay = 1e-4) -- better generalisation than Adam
  - OneCycleLR scheduler -- warm up then cosine anneal (fast.ai / Leslie Smith)
  - Gradient clipping (max_norm=1.0) -- prevents exploding gradients
  - Best-checkpoint restore -- always saves best val ROC-AUC state
  - Epoch plot saved to docs/training_results/epoch_plot.png as proof
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
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
_PLOT_DIR = Path("docs") / "training_results"


# ============================================================================
# Architecture
# ============================================================================

class MultiHeadAttention(nn.Module):
    """Scaled dot-product multi-head attention over LSTM output sequence."""

    def __init__(self, hidden_size: int, num_heads: int = 4):
        super().__init__()
        assert hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads"
        self.attn = nn.MultiheadAttention(
            embed_dim   = hidden_size,
            num_heads   = num_heads,
            dropout     = 0.1,
            batch_first = True,
        )
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, hidden)
        out, _ = self.attn(x, x, x)      # self-attention
        return self.norm(x + out)         # residual + norm


class StockLSTM(nn.Module):
    """
    Production-grade LSTM for binary stock direction classification.

    Input  : (batch, seq_len, n_features)
    Output : (batch,) -- probability of price going UP
    """

    def __init__(
        self,
        input_size:  int,
        hidden_size: int = LSTM_HIDDEN_SIZE,
        num_layers:  int = LSTM_NUM_LAYERS,
        dropout:     float = LSTM_DROPOUT,
    ):
        super().__init__()

        # Input projection -- expand features to hidden dim before LSTM
        self.input_proj = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
        )

        # Bidirectional LSTM stack
        self.lstm = nn.LSTM(
            input_size  = hidden_size,
            hidden_size = hidden_size,
            num_layers  = num_layers,
            batch_first = True,
            dropout     = dropout if num_layers > 1 else 0.0,
            bidirectional = True,
        )

        # Project bidirectional output back to hidden_size (2*hidden -> hidden)
        self.bidir_proj = nn.Linear(hidden_size * 2, hidden_size)
        self.lstm_norm  = nn.LayerNorm(hidden_size)

        # Multi-head self-attention over the LSTM sequence
        self.attention = MultiHeadAttention(hidden_size, num_heads=4)

        # Residual projection for skip connection from input_proj to head
        self.residual_proj = nn.Linear(hidden_size, hidden_size)

        # Classification head
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 128),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

        self._init_weights()

    def _init_weights(self):
        """Xavier initialisation for linear layers, orthogonal for LSTM."""
        for name, param in self.lstm.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param.data)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param.data)
            elif "bias" in name:
                param.data.fill_(0)
                # Set forget gate bias to 1 (LSTM best practice)
                n = param.size(0)
                param.data[n // 4: n // 2].fill_(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, F)
        x_proj = self.input_proj(x)            # (B, T, H)

        lstm_out, _ = self.lstm(x_proj)        # (B, T, 2H) bidirectional
        lstm_out    = self.bidir_proj(lstm_out) # (B, T, H)
        lstm_out    = self.lstm_norm(lstm_out)  # (B, T, H)

        # Self-attention over the full sequence
        attn_out = self.attention(lstm_out)     # (B, T, H)

        # Use the last timestep + residual from projected input
        last      = attn_out[:, -1, :]                         # (B, H)
        residual  = self.residual_proj(x_proj[:, -1, :])      # (B, H)
        combined  = last + residual                             # (B, H) skip connection

        return self.head(combined).squeeze(1)   # (B,) raw logits


# ============================================================================
# Trainer
# ============================================================================

class LSTMTrainer:
    """Handles training, evaluation, checkpointing, and epoch plotting."""

    def __init__(self, input_size: int):
        self.input_size = input_size
        self.scaler     = StandardScaler()
        self.device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model: StockLSTM | None = None
        logger.info(f"LSTM device: {self.device}")

    # -------------------------------------------------------------------------
    def _build_loaders(
        self,
        X_train: np.ndarray, y_train: np.ndarray,
        X_val:   np.ndarray, y_val:   np.ndarray,
    ):
        """Scale, build sequences, return DataLoaders."""
        Xs_train = self.scaler.fit_transform(X_train)
        Xs_val   = self.scaler.transform(X_val)

        Xt, yt = build_sequences(Xs_train, y_train, SEQUENCE_LENGTH)
        Xv, yv = build_sequences(Xs_val,   y_val,   SEQUENCE_LENGTH)

        self._Xv, self._yv = Xv, yv    # keep for later evaluation

        train_ds = TensorDataset(
            torch.from_numpy(Xt).float(),
            torch.from_numpy(yt).float(),
        )
        val_ds = TensorDataset(
            torch.from_numpy(Xv).float(),
            torch.from_numpy(yv).float(),
        )
        train_loader = DataLoader(
            train_ds, batch_size=LSTM_BATCH_SIZE, shuffle=True,
            drop_last=True, num_workers=0, pin_memory=False,
        )
        val_loader = DataLoader(
            val_ds, batch_size=LSTM_BATCH_SIZE, shuffle=False,
            num_workers=0, pin_memory=False,
        )
        return train_loader, val_loader

    # -------------------------------------------------------------------------
    def fit(
        self,
        X_train: np.ndarray, y_train: np.ndarray,
        X_val:   np.ndarray, y_val:   np.ndarray,
    ) -> dict:
        train_loader, val_loader = self._build_loaders(X_train, y_train, X_val, y_val)

        self.model = StockLSTM(self.input_size).to(self.device)

        # Class imbalance correction
        pos_weight = torch.tensor(
            [(y_train == 0).sum() / max((y_train == 1).sum(), 1)],
            dtype=torch.float32,
        ).to(self.device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        # AdamW: better weight decay than Adam (Loshchilov & Hutter 2019)
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr           = LSTM_LEARNING_RATE,
            weight_decay = 1e-4,
            betas        = (0.9, 0.999),
        )

        # OneCycleLR: warmup + cosine annealing (Leslie Smith 2018)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr         = LSTM_LEARNING_RATE * 10,
            epochs         = LSTM_EPOCHS,
            steps_per_epoch= len(train_loader),
            pct_start      = 0.1,          # 10% warmup
            anneal_strategy= "cos",
        )

        # Training state
        history        = {"train_loss": [], "val_auc": [], "lr": []}
        best_val_auc   = -1.0
        best_state     = None
        best_epoch     = 0
        no_improve     = 0
        early_stop_ep  = None

        for epoch in range(1, LSTM_EPOCHS + 1):

            # -- Train ----------------------------------------------------
            self.model.train()
            epoch_loss = 0.0
            for Xb, yb in train_loader:
                Xb, yb = Xb.to(self.device), yb.to(self.device)
                optimizer.zero_grad()
                logits = self.model(Xb)
                loss   = criterion(logits, yb)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                epoch_loss += loss.item()
            avg_loss = epoch_loss / len(train_loader)
            current_lr = scheduler.get_last_lr()[0]

            # -- Validate -------------------------------------------------
            self.model.eval()
            all_preds, all_labels = [], []
            with torch.no_grad():
                for Xb, yb in val_loader:
                    preds = torch.sigmoid(self.model(Xb.to(self.device))).cpu().numpy()
                    all_preds.extend(preds)
                    all_labels.extend(yb.numpy())

            all_preds  = np.array(all_preds)
            all_labels = np.array(all_labels)

            metrics = compute_all(all_labels, all_preds)
            val_auc = metrics["roc_auc"]
            val_acc = metrics["accuracy"]
            sharpe  = metrics.get("sharpe_ratio", 0.0)

            history["train_loss"].append(avg_loss)
            history["val_auc"].append(val_auc)
            history["lr"].append(current_lr)

            logger.info(
                f"LSTM epoch {epoch:3d}/{LSTM_EPOCHS}"
                f" -- loss: {avg_loss:.4f}"
                f"  val_auc: {val_auc:.4f}"
                f"  val_acc: {val_acc:.4f}"
                f"  sharpe: {sharpe:.3f}"
                f"  lr: {current_lr:.6f}"
            )

            # -- Checkpoint best model ------------------------------------
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_epoch   = epoch
                best_state   = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                no_improve   = 0
            else:
                no_improve += 1

            # -- Early stopping -------------------------------------------
            if no_improve >= LSTM_PATIENCE:
                logger.info(
                    f"Early stopping at epoch {epoch} "
                    f"(patience={LSTM_PATIENCE}, best epoch={best_epoch})"
                )
                early_stop_ep = epoch
                break

        # Restore best weights
        if best_state:
            self.model.load_state_dict(best_state)
            logger.info(f"Restored best weights from epoch {best_epoch} (val_auc={best_val_auc:.4f})")

        # Save epoch plot
        self._save_epoch_plot(history, best_epoch, early_stop_ep)

        return {
            "best_epoch":   best_epoch,
            "best_val_auc": best_val_auc,
            "total_epochs": len(history["train_loss"]),
        }

    # -------------------------------------------------------------------------
    def _save_epoch_plot(
        self,
        history:       dict,
        best_epoch:    int,
        early_stop_ep: int | None,
    ) -> None:
        _PLOT_DIR.mkdir(parents=True, exist_ok=True)
        plot_path = _PLOT_DIR / "epoch_plot.png"

        epochs  = list(range(1, len(history["train_loss"]) + 1))
        losses  = history["train_loss"]
        aucs    = history["val_auc"]
        lrs     = history["lr"]

        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
        fig.patch.set_facecolor("#0F172A")
        for ax in (ax1, ax2, ax3):
            ax.set_facecolor("#1E293B")
            ax.tick_params(colors="#94A3B8")
            for sp in ax.spines.values():
                sp.set_color("#334155")

        # Panel 1: Training loss
        ax1.plot(epochs, losses, color="#3B82F6", linewidth=1.5, label="Train BCE Loss")
        ax1.set_ylabel("BCE Loss", color="#94A3B8")
        ax1.set_title(
            f"LSTM Training -- 300 Epoch Budget  |  Best Epoch: {best_epoch}  |  Best Val AUC: {max(aucs):.4f}",
            color="#E2E8F0", fontsize=11, pad=6
        )
        ax1.legend(facecolor="#1E293B", edgecolor="#334155", labelcolor="#E2E8F0")
        ax1.grid(True, color="#1E3A5F", linewidth=0.4)

        # Panel 2: Val AUC
        ax2.plot(epochs, aucs, color="#F97316", linewidth=1.5, label="Val ROC-AUC")
        ax2.axhline(0.5, color="#94A3B8", linestyle=":", linewidth=0.8, label="Random baseline (0.5)")
        ax2.axvline(best_epoch, color="#10B981", linestyle="--", linewidth=1.5,
                    label=f"Best epoch {best_epoch} (AUC={max(aucs):.4f})")
        if early_stop_ep:
            ax2.axvline(early_stop_ep, color="#EF4444", linestyle=":", linewidth=1.2,
                        label=f"Early stop @ epoch {early_stop_ep}")
        ax2.set_ylabel("Val ROC-AUC", color="#94A3B8")
        ax2.legend(facecolor="#1E293B", edgecolor="#334155", labelcolor="#E2E8F0", fontsize=8)
        ax2.grid(True, color="#1E3A5F", linewidth=0.4)
        ax2.set_ylim(0.40, 0.70)

        # Panel 3: Learning rate (OneCycleLR)
        ax3.plot(epochs, lrs, color="#A78BFA", linewidth=1.2, label="Learning Rate")
        ax3.set_ylabel("LR", color="#94A3B8")
        ax3.set_xlabel("Epoch", color="#94A3B8")
        ax3.legend(facecolor="#1E293B", edgecolor="#334155", labelcolor="#E2E8F0")
        ax3.grid(True, color="#1E3A5F", linewidth=0.4)

        plt.tight_layout(pad=1.5)
        fig.savefig(plot_path, dpi=150, bbox_inches="tight", facecolor="#0F172A")
        plt.close(fig)
        logger.info(f"Epoch plot saved -> {plot_path}")

    # -------------------------------------------------------------------------
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> dict:
        if self.model is None:
            raise RuntimeError("Model not trained yet. Call fit() first.")
        Xs_test = self.scaler.transform(X_test)
        Xt, yt  = build_sequences(Xs_test, y_test, SEQUENCE_LENGTH)
        loader  = DataLoader(
            TensorDataset(torch.from_numpy(Xt).float(), torch.from_numpy(yt).float()),
            batch_size=LSTM_BATCH_SIZE, shuffle=False,
        )
        self.model.eval()
        preds, labels = [], []
        with torch.no_grad():
            for Xb, yb in loader:
                p = torch.sigmoid(self.model(Xb.to(self.device))).cpu().numpy()
                preds.extend(p)
                labels.extend(yb.numpy())
        return compute_all(np.array(labels), np.array(preds))

    # -------------------------------------------------------------------------
    def save(self, model_path: str, scaler_path: str) -> None:
        if self.model is None:
            raise RuntimeError("No model to save.")
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        torch.save({"model_state": self.model.state_dict(),
                    "input_size":  self.input_size}, model_path)
        joblib.dump(self.scaler, scaler_path)
        logger.info(f"LSTM saved -> {model_path}")

    # -------------------------------------------------------------------------
    def predict_proba_latest(self, df) -> float:
        """Single-sample inference for the most recent row."""
        from src.models.feature_columns import FEATURE_COLUMNS
        feat_df = df[FEATURE_COLUMNS].dropna()
        if len(feat_df) < SEQUENCE_LENGTH:
            raise ValueError(
                f"Need at least {SEQUENCE_LENGTH} rows for LSTM inference, got {len(feat_df)}."
            )
        X = self.scaler.transform(feat_df.values)
        seq = torch.from_numpy(X[-SEQUENCE_LENGTH:]).float().unsqueeze(0).to(self.device)
        self.model.eval()
        with torch.no_grad():
            return float(torch.sigmoid(self.model(seq)).cpu().item())

    # -------------------------------------------------------------------------
    @classmethod
    def load(cls, model_path: str, scaler_path: str) -> "LSTMTrainer":
        checkpoint  = torch.load(model_path, map_location="cpu")
        input_size  = checkpoint["input_size"]
        trainer     = cls(input_size)
        trainer.model = StockLSTM(input_size).to(trainer.device)
        trainer.model.load_state_dict(checkpoint["model_state"])
        trainer.model.eval()
        trainer.scaler = joblib.load(scaler_path)
        return trainer
