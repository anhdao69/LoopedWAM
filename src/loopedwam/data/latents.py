"""Disk cache of the frozen VAE latents of every training clip (BF16 bits in a uint16 memmap).

Window i of a prepared dataset owns row i. A row is valid once its uint8 flag is set, which happens only after the
payload has been flushed, so an interrupted precompute never leaves a partially written valid row.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch


class LatentCache:
    def __init__(self, directory, size: int, latent_shape, *, writable: bool = False):
        self.directory = Path(directory)
        self.size, self.latent_shape = int(size), tuple(int(n) for n in latent_shape)
        data, valid, meta = (self.directory / n for n in ('latents.uint16', 'valid.uint8', 'metadata.json'))
        nbytes = self.size * int(np.prod(self.latent_shape)) * 2
        if writable and not meta.exists():
            self.directory.mkdir(parents=True, exist_ok=True)
            with data.open('xb') as stream:
                stream.truncate(nbytes)
            with valid.open('xb') as stream:
                stream.truncate(self.size)
            meta.write_text(json.dumps(dict(size=self.size, latent_shape=list(self.latent_shape),
                                            storage='bfloat16 bits as uint16')))
        info = json.loads(meta.read_text())
        if info['size'] != self.size or list(info['latent_shape']) != list(self.latent_shape):
            raise ValueError(f'{self.directory}: cache holds {info["size"]} x {info["latent_shape"]}, '
                             f'expected {self.size} x {list(self.latent_shape)}')
        if data.stat().st_size != nbytes or valid.stat().st_size != self.size:
            raise ValueError(f'{self.directory}: cache files have unexpected sizes')
        mode = 'r+' if writable else 'r'
        self.latents = np.memmap(data, mode=mode, dtype=np.uint16, shape=(self.size, *self.latent_shape))
        self.valid = np.memmap(valid, mode=mode, dtype=np.uint8, shape=(self.size,))

    def complete(self) -> bool:
        return bool(np.all(self.valid == 1))

    def read_bits(self, indices, frames=None) -> torch.Tensor:
        """int16 view of the BF16 bits [B, *latent_shape] (or only the first `frames` latent frames)."""
        ids = np.asarray(indices, dtype=np.int64)
        if not np.all(self.valid[ids] == 1):
            raise RuntimeError('Latent cache miss: run scripts/precompute_latents.py first')
        rows = self.latents[ids] if frames is None else self.latents[ids, :, :frames]
        return torch.from_numpy(np.ascontiguousarray(rows).view(np.int16))

    def read(self, indices, frames=None) -> torch.Tensor:
        """FP32 latents [B, *latent_shape]."""
        return self.read_bits(indices, frames).view(torch.bfloat16).float()

    def write(self, indices, latents: torch.Tensor):
        """Store FP32 latents that are exact BF16 values (the VAE runs in BF16)."""
        bits = latents.to(torch.bfloat16)
        if not torch.equal(bits.float(), latents.float()) or not torch.isfinite(latents).all():
            raise ValueError('Latents must be finite BF16 values')
        ids = np.asarray(indices, dtype=np.int64)
        self.latents[ids] = bits.cpu().contiguous().view(torch.int16).numpy().view(np.uint16)
        self.latents.flush()
        self.valid[ids] = 1
        self.valid.flush()
