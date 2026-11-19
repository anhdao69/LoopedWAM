"""Per-dimension linear normalization of actions and proprioception (the FastWAM normalizer)."""
from __future__ import annotations

import json
from pathlib import Path

import torch

MODES = {'libero': 'min/max', 'robotwin': 'z-score', 'robocasa_gr1': 'min/max'}


class Normalizer:
    """`min/max` maps [min, max] to [-1, 1] (dimensions with max - min < 1e-4 are only shifted by -min);
    `z-score` maps to zero mean and unit variance. Outputs are clamped to [-5, 5]."""

    std_reg = 1e-8
    range_tol = 1e-4

    def __init__(self, stats: dict, mode: str):
        if mode == 'z-score':
            self.scale = 1.0 / (stats['std'] + self.std_reg)
            self.offset = -stats['mean'] / (stats['std'] + self.std_reg)
        elif mode == 'min/max':
            low, high = stats['min'], stats['max']
            span = high - low
            ignore = span < self.range_tol
            span[ignore] = 2.0
            self.scale = 2.0 / span
            self.offset = -1.0 - self.scale * low
            self.offset[ignore] = 0.0 - low[ignore]
        else:
            raise ValueError(f'Unknown normalization mode {mode!r}')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.clamp(x * self.scale + self.offset, -5.0, 5.0)

    def backward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.offset.to(x.device)) / self.scale.to(x.device)


def load_normalizers(stats, mode: str) -> dict:
    """{'action': Normalizer, 'state': Normalizer} from a dataset_stats.json (path or loaded dict)."""
    if not isinstance(stats, dict):
        stats = json.loads(Path(stats).read_text())
    return {kind: Normalizer({key.removeprefix('global_'): torch.tensor(value, dtype=torch.float32)
                              for key, value in stats[kind]['default'].items() if key.startswith('global_')}, mode)
            for kind in ('action', 'state')}
