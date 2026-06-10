import logging
import os

import joblib
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from src.config.ml_config import (
    BEST_LSTM_PATH,
    FEATURE_IMPORTANCE_PATH,
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

logger = logging.getLogger(__name__)


class StockLSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, dropout: float):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        out = self.dropout(out[:, -1, :])
        return self.fc(out)


class LSTMTrainer:
    """Train, evaluate, save, and load an LSTM direction classifier."""

    def __init__(
        self,
        seq_len: int = SEQUENCE_LENGTH,
        hidden_size: int = LSTM_HIDDEN_SIZE,
        num_layers: int = LSTM_NUM_LAYERS,
        dropout: float = LSTM_DROPOUT,
    ):
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.scaler = StandardScaler()
        self.model: StockLSTM | None = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _scale_sequences(self, x: np.ndarray, fit: bool = False) -> np.ndarray:
        n, t, f = x.shape
        flat = x.reshape(-1, f)
        if fit:
            scaled = self.scaler.fit_transform(flat)
        else:
            scaled = self.scaler.transform(flat)
        return scaled.reshape(n, t, f).astype(np.float32)

    def _to_loader(self, x: np.ndarray, y: np.ndarray, shuffle: bool) -> DataLoader:
        dataset = TensorDataset(
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32),
        )
        return DataLoader(dataset, batch_size=LSTM_BATCH_SIZE, shuffle=shuffle)

    def fit(self, train_df, val_df) -> dict:
        x_train, y_train = build_sequences(train_df, self.seq_len)
        x_val, y_val = build_sequences(val_df, self.seq_len)

        x_train = self._scale_sequences(x_train, fit=True)
        x_val = self._scale_sequences(x_val, fit=False)

        train_loader = self._to_loader(x_train, y_train, shuffle=True)
        val_loader = self._to_loader(x_val, y_val, shuffle=False)

        input_size = len(FEATURE_COLUMNS)
        self.model = StockLSTM(
            input_size=input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
        ).to(self.device)

        pos = y_train.sum()
        neg = len(y_train) - pos
        pos_weight = torch.tensor([neg / max(pos, 1)], device=self.device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=LSTM_LEARNING_RATE)

        best_val_auc = -1.0
        patience_counter = 0
        best_state = None

        for epoch in range(1, LSTM_EPOCHS + 1):
            self.model.train()
            for xb, yb in train_loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                optimizer.zero_grad()
                logits = self.model(xb).squeeze(-1)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()

            val_metrics = self.evaluate_sequences(x_val, y_val)
            logger.info(
                "LSTM epoch %d — val loss proxy auc=%.4f acc=%.4f",
                epoch, val_metrics["roc_auc"], val_metrics["accuracy"],
            )

            if val_metrics["roc_auc"] > best_val_auc:
                best_val_auc = val_metrics["roc_auc"]
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= LSTM_PATIENCE:
                    logger.info("Early stopping at epoch %d", epoch)
                    break

        if best_state:
            self.model.load_state_dict(best_state)

        return {"best_val_roc_auc": best_val_auc}

    def evaluate_sequences(self, x: np.ndarray, y: np.ndarray) -> dict:
        if self.model is None:
            raise RuntimeError("Model not trained.")

        self.model.eval()
        loader = self._to_loader(x, y, shuffle=False)
        probas: list[float] = []

        with torch.no_grad():
            for xb, _ in loader:
                xb = xb.to(self.device)
                logits = self.model(xb).squeeze(-1)
                batch_proba = torch.sigmoid(logits).cpu().numpy()
                probas.extend(batch_proba.tolist())

        probas_arr = np.array(probas)
        preds = (probas_arr >= 0.5).astype(int)

        return {
            "accuracy": float(accuracy_score(y, preds)),
            "precision": float(precision_score(y, preds, zero_division=0)),
            "recall": float(recall_score(y, preds, zero_division=0)),
            "f1": float(f1_score(y, preds, zero_division=0)),
            "roc_auc": float(roc_auc_score(y, probas_arr)),
        }

    def evaluate_df(self, df) -> dict:
        x, y = build_sequences(df, self.seq_len)
        x = self._scale_sequences(x, fit=False)
        return self.evaluate_sequences(x, y)

    def predict_proba_latest(self, df) -> float:
        if self.model is None:
            raise RuntimeError("Model not trained or loaded.")

        feature_df = df[FEATURE_COLUMNS].dropna()
        if len(feature_df) < self.seq_len:
            raise ValueError(
                f"Need at least {self.seq_len} rows of feature history for LSTM prediction."
            )

        seq = feature_df.iloc[-self.seq_len :][FEATURE_COLUMNS].values.astype(np.float32)
        seq = self.scaler.transform(seq).reshape(1, self.seq_len, -1)

        self.model.eval()
        with torch.no_grad():
            x = torch.tensor(seq, dtype=torch.float32).to(self.device)
            logit = self.model(x).squeeze(-1)
            return float(torch.sigmoid(logit).cpu().item())

    def save(self, model_path: str = BEST_LSTM_PATH, scaler_path: str = LSTM_SCALER_PATH) -> None:
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        torch.save({
            "state_dict": self.model.state_dict(),
            "seq_len": self.seq_len,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
            "input_size": len(FEATURE_COLUMNS),
        }, model_path)
        joblib.dump(self.scaler, scaler_path)

    @classmethod
    def load(
        cls,
        model_path: str = BEST_LSTM_PATH,
        scaler_path: str = LSTM_SCALER_PATH,
    ) -> "LSTMTrainer":
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
        trainer = cls(
            seq_len=checkpoint["seq_len"],
            hidden_size=checkpoint["hidden_size"],
            num_layers=checkpoint["num_layers"],
            dropout=checkpoint["dropout"],
        )
        trainer.scaler = joblib.load(scaler_path)
        trainer.model = StockLSTM(
            input_size=checkpoint["input_size"],
            hidden_size=checkpoint["hidden_size"],
            num_layers=checkpoint["num_layers"],
            dropout=checkpoint["dropout"],
        )
        trainer.model.load_state_dict(checkpoint["state_dict"])
        trainer.model.to(trainer.device)
        trainer.model.eval()
        return trainer
