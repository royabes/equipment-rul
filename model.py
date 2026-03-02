# model.py — PyTorch LSTM for Remaining Useful Life prediction
import torch
import torch.nn as nn


class RUL_LSTM(nn.Module):
    """
    2-layer LSTM for RUL regression.
    Input:  (batch, window=30, features=14)
    Output: (batch, 1) — predicted RUL in cycles
    """
    def __init__(self, input_size: int = 14, hidden_size: int = 64, num_layers: int = 2,
                 dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0.0
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (h_n, _) = self.lstm(x)
        return self.fc(self.dropout(h_n[-1]))


def load_model(path: str, input_size: int = 14, device: str = "cpu") -> RUL_LSTM:
    model = RUL_LSTM(input_size=input_size)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model
