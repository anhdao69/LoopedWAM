"""Which windows each optimizer update uses, and the noise of every training window.

* `random_batches`: every epoch is a fresh permutation (seed + epoch) cut into global batches; each global batch is
  split evenly across ranks. Used by every LoopedWAM stage except LIBERO stage 2.
* `TraversalPlan`: LIBERO stage 2 visits each demonstration along strided traversals (window u, u + 10, u + 20, ... at
  the 10-step replanning stride). 32 streams take traversals in a fixed shuffled order and contribute 4 consecutive
  windows each per update (32 x 4 = 128 windows); rank r owns streams r, r + world, ...
* `elastic_batches`: the LoopedWAM-E sampler (permutation per epoch, padded to whole global batches).
"""
from __future__ import annotations

import hashlib
import math

import numpy as np
import torch


# ---------------------------------------------------------------------- LoopedWAM: random epochs
def updates_per_epoch(size: int, global_batch: int) -> int:
    return math.ceil(size / global_batch)


def random_batches(size, global_batch, world, rank, seed, epoch, start=0):
    """Yield (global batch size, this rank's window indices) for each update of `epoch`, from batch `start`."""
    order = torch.randperm(size, generator=torch.Generator().manual_seed(seed + epoch)).numpy()
    for lo in range(start * global_batch, size, global_batch):
        chunk = order[lo:lo + global_batch]
        q, r = divmod(len(chunk), world)
        first = rank * q + min(rank, r)
        yield len(chunk), chunk[first:first + q + int(rank < r)].tolist()


def slot_seed(seed, a, b, c) -> int:
    """64-bit noise seed of one training window, keyed by (seed, a, b, c)."""
    digest = hashlib.sha256(f'{seed}:{a}:{b}:{c}'.encode()).digest()
    return int.from_bytes(digest[:8], 'little') & ((1 << 63) - 1)


def slot_noise(model, seeds, latent_shape, device):
    """Video/action noise and flow-matching timesteps, one CUDA generator per window seed."""
    nv, na, uv, ua = [], [], [], []
    for s in seeds:
        g = torch.Generator(device=device).manual_seed(s)
        nv.append(torch.randn(latent_shape, generator=g, device=device))
        na.append(torch.randn((32, model.action_dim), generator=g, device=device))
        u = torch.rand((2,), generator=g, device=device)
        uv.append(u[0])
        ua.append(u[1])
    sv, sa = model.train_video_scheduler, model.train_action_scheduler
    return dict(video=torch.stack(nv), action=torch.stack(na),
                t_video=sv._phi(torch.stack(uv), sv.shift) * float(sv.num_train_timesteps),
                t_action=sa._phi(torch.stack(ua), sa.shift) * float(sa.num_train_timesteps))


# ---------------------------------------------------------------------- LoopedWAM LIBERO stage 2: traversals
def build_traversals(lengths, epochs: int, seed: int, stride: int = 10) -> np.ndarray:
    """Rows (epoch, demonstration, phase u) covering every (d, u < stride) once per epoch, shuffled per epoch."""
    rows = []
    for epoch in range(epochs):
        pairs = np.asarray([(d, u) for d in range(len(lengths)) for u in range(stride) if u < lengths[d]],
                           dtype=np.int64)
        order = np.random.default_rng(seed + epoch).permutation(len(pairs))
        rows.append(np.column_stack([np.full(len(pairs), epoch), pairs[order]]))
    return np.concatenate(rows).astype(np.int64)


class TraversalPlan:
    """Stream schedule over the traversals; world-size independent and resumable (`state_dict`)."""

    def __init__(self, starts, lengths, epochs, seed, streams=32, windows_per_stream=4, stride=10):
        self.starts, self.lengths = np.asarray(starts, np.int64), np.asarray(lengths, np.int64)
        self.traversals = build_traversals(self.lengths, epochs, seed, stride)
        self.streams, self.per_stream, self.stride = streams, windows_per_stream, stride
        self.cursor = 0
        self.state = [[-1, 0] for _ in range(streams)]   # (traversal row or -1, next query index)

    def _length(self, d, u):
        return max(0, (int(self.lengths[d]) - u + self.stride - 1) // self.stride)

    @property
    def exhausted(self):
        return self.cursor >= len(self.traversals) and all(t < 0 for t, _ in self.state)

    def next_update(self):
        """slots[k][w] = global window index of stream k's w-th window this update, or None."""
        slots = []
        for st in self.state:
            if st[0] < 0 and self.cursor < len(self.traversals):
                st[0], st[1] = self.cursor, 0
                self.cursor += 1
            if st[0] < 0:
                slots.append([None] * self.per_stream)
                continue
            _, d, u = (int(v) for v in self.traversals[st[0]])
            n = self._length(d, u)
            slots.append([int(self.starts[d]) + u + self.stride * (st[1] + w) if st[1] + w < n else None
                          for w in range(self.per_stream)])
            st[1] += self.per_stream
            if st[1] >= n:
                st[0] = -1
        return slots

    def total_updates(self) -> int:
        copy = TraversalPlan.__new__(TraversalPlan)
        copy.__dict__.update(self.__dict__, cursor=0, state=[[-1, 0] for _ in range(self.streams)])
        updates = 0
        while not copy.exhausted:
            copy.next_update()
            updates += 1
        return updates

    def state_dict(self):
        return dict(cursor=self.cursor, state=[list(s) for s in self.state])

    def load_state_dict(self, state):
        self.cursor, self.state = int(state['cursor']), [list(s) for s in state['state']]


# ---------------------------------------------------------------------- LoopedWAM-E
def elastic_batches(size, microbatch, world, rank, seed, epoch, start=0):
    """Yield this rank's `microbatch` window indices per microstep (-1 = padding with zero weight)."""
    order = torch.randperm(size, generator=torch.Generator().manual_seed(seed + epoch)).tolist()
    unit = microbatch * world
    order += [-1] * ((-len(order)) % unit)
    for lo in range(start * unit, len(order), unit):
        yield order[lo + rank * microbatch:lo + (rank + 1) * microbatch]


def elastic_noise(indices, epoch, seed, action_dim):
    """Per-window CPU noise: the initial action noise, the anchor noise and the anchor time."""
    noise, anchor, times = [], [], []
    for index in indices:
        g = torch.Generator().manual_seed(seed + 1000003 * epoch + 97 * (max(int(index), 0) + 1))
        noise.append(torch.randn(32, action_dim, generator=g))
        anchor.append(torch.randn(32, action_dim, generator=g))
        times.append(torch.rand(1, generator=g))
    return torch.stack(noise), torch.stack(anchor), torch.cat(times)
