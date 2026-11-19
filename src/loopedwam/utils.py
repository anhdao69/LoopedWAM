"""Small shared helpers."""
from __future__ import annotations

import hashlib
import logging
import math
from pathlib import Path


def get_logger(name: str = __name__, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)
    return logger


def sha256_file(path) -> str:
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(16 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def lr_factor(update: int, total: int, warmup: int) -> float:
    """Linear warmup over `warmup` updates, then cosine decay to 1% of the peak at `total`."""
    if update < warmup:
        return (update + 1) / max(warmup, 1)
    progress = min(1.0, (update - warmup) / max(total - warmup - 1, 1))
    return 0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * progress))


def atomic_save(payload, path):
    import torch
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    torch.save(payload, tmp)
    tmp.replace(path)
