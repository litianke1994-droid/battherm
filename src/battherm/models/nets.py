"""PyTorch forecasters. Both map (past window, future load) -> temperature rise over H steps.

Design choices worth explaining in an interview:
* Encoder-decoder: the past window summarises the thermal state; the decoder consumes the
  future current step by step, like integrating an ODE.
* Increment head + cumsum: the network predicts per-step increments dT_h and the output is
  their cumulative sum, so the forecast is a trajectory, not H independent numbers.
* Causality along the horizon: the forecast for step h may only use future inputs <= h
  (LSTM decoder is causal by construction; the CNN uses left-padded causal convolutions).
"""

from __future__ import annotations

import torch
from torch import nn


class EncoderDecoderLSTM(nn.Module):
    def __init__(self, n_past: int, n_future: int, hidden: int = 64, layers: int = 1, dropout: float = 0.0):
        super().__init__()
        drop = dropout if layers > 1 else 0.0
        self.encoder = nn.LSTM(n_past, hidden, layers, batch_first=True, dropout=drop)
        self.decoder = nn.LSTM(n_future, hidden, layers, batch_first=True, dropout=drop)
        self.head = nn.Linear(hidden, 1)

    def forward(self, past: torch.Tensor, future: torch.Tensor) -> torch.Tensor:
        _, state = self.encoder(past)  # (h, c) of the last past step
        out, _ = self.decoder(future, state)  # (B, H, hidden)
        return torch.cumsum(self.head(out).squeeze(-1), dim=1)  # (B, H)


class CausalConv1d(nn.Module):
    def __init__(self, c_in: int, c_out: int, kernel: int, dilation: int = 1):
        super().__init__()
        self.pad = (kernel - 1) * dilation
        self.conv = nn.Conv1d(c_in, c_out, kernel, dilation=dilation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (B, C, L)
        return self.conv(nn.functional.pad(x, (self.pad, 0)))


class CNNForecaster(nn.Module):
    """Dilated causal CNN encoder over the past + causal CNN decoder over the horizon."""

    def __init__(self, n_past: int, n_future: int, channels: int = 32, kernel: int = 5, enc_layers: int = 4):
        super().__init__()
        layers, c = [], n_past
        for i in range(enc_layers):
            layers += [CausalConv1d(c, channels, kernel, dilation=2**i), nn.GELU()]
            c = channels
        self.encoder = nn.Sequential(*layers)
        self.context = nn.Linear(2 * channels, channels)
        self.decoder = nn.Sequential(
            CausalConv1d(n_future + channels, channels, kernel), nn.GELU(),
            CausalConv1d(channels, channels, kernel, dilation=2), nn.GELU(),
        )
        self.head = nn.Conv1d(channels, 1, 1)

    def forward(self, past: torch.Tensor, future: torch.Tensor) -> torch.Tensor:
        z = self.encoder(past.transpose(1, 2))  # (B, C, W)
        ctx = self.context(torch.cat([z[:, :, -1], z.mean(dim=2)], dim=1))  # last step + average
        H = future.shape[1]
        dec_in = torch.cat([future.transpose(1, 2), ctx[:, :, None].expand(-1, -1, H)], dim=1)
        inc = self.head(self.decoder(dec_in)).squeeze(1)  # (B, H)
        return torch.cumsum(inc, dim=1)


def build_model(kind: str, n_past: int, n_future: int, cfg: dict) -> nn.Module:
    if kind in ("lstm", "hybrid_lstm"):
        c = cfg.get("lstm", {})
        return EncoderDecoderLSTM(n_past, n_future, c.get("hidden", 64), c.get("layers", 1), c.get("dropout", 0.0))
    if kind == "cnn":
        c = cfg.get("cnn", {})
        return CNNForecaster(n_past, n_future, c.get("channels", 32), c.get("kernel", 5), c.get("enc_layers", 4))
    raise ValueError(f"unknown model kind: {kind}")
