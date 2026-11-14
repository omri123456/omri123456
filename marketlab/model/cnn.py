"""Multi-branch temporal CNN.

Each kind of information (market, rates, macro + forecasts, news + events, calendar) gets its
own stack of causal dilated 1-D convolutions over the last `window` trading days. The branch
summaries are concatenated and feed two small heads per horizon: the probability that the
index rises, and the expected volatility-scaled return.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalConv(nn.Module):
    """Conv1d that only looks backwards in time (left padding)."""

    def __init__(self, channels: int, kernel: int, dilation: int, dropout: float):
        super().__init__()
        self.pad = (kernel - 1) * dilation
        self.conv = nn.Conv1d(channels, channels, kernel, dilation=dilation)
        self.norm = nn.BatchNorm1d(channels)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(F.pad(x, (self.pad, 0)))
        return x + self.drop(F.gelu(self.norm(out)))  # residual


class Branch(nn.Module):
    def __init__(self, in_channels: int, hidden: int, dropout: float, channel_dropout: float = 0.15):
        super().__init__()
        # Dropping whole input series at random keeps the model from leaning on any single one.
        self.channel_drop = nn.Dropout1d(channel_dropout)
        self.mix = nn.Conv1d(in_channels, hidden, kernel_size=1)
        self.blocks = nn.Sequential(
            CausalConv(hidden, 5, 1, dropout),
            CausalConv(hidden, 5, 2, dropout),
            CausalConv(hidden, 5, 4, dropout),
        )
        self.out_features = 3 * hidden

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.blocks(F.gelu(self.mix(self.channel_drop(x))))
        # the most recent state, plus what happened on average and at the extreme in the window
        return torch.cat([h[:, :, -1], h.mean(dim=-1), h.amax(dim=-1)], dim=1)


class MarketCNN(nn.Module):
    def __init__(self, branch_index: dict[str, list[int]], n_horizons: int, hidden: int = 16,
                 dropout: float = 0.25, head: int = 48):
        super().__init__()
        self.branch_names = [name for name, idx in branch_index.items() if idx]
        self.branches = nn.ModuleDict({name: Branch(len(branch_index[name]), hidden, dropout)
                                       for name in self.branch_names})
        for name in self.branch_names:
            self.register_buffer(f"idx_{name}", torch.tensor(branch_index[name], dtype=torch.long))
        fused = sum(self.branches[name].out_features for name in self.branch_names)
        self.trunk = nn.Sequential(nn.Dropout(dropout), nn.Linear(fused, head), nn.GELU(), nn.Dropout(dropout))
        self.direction = nn.Linear(head, n_horizons)
        self.magnitude = nn.Linear(head, n_horizons)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        parts = [self.branches[name](x.index_select(1, getattr(self, f"idx_{name}")))
                 for name in self.branch_names]
        h = self.trunk(torch.cat(parts, dim=1))
        return self.direction(h), self.magnitude(h)


class LinearBaseline(nn.Module):
    """Reference model: a linear map of the latest feature row only (no history, no interactions)."""

    def __init__(self, n_channels: int, n_horizons: int):
        super().__init__()
        self.direction = nn.Linear(n_channels, n_horizons)
        self.magnitude = nn.Linear(n_channels, n_horizons)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        last = x[:, :, -1]
        return self.direction(last), self.magnitude(last)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
