"""Training windows: every frame of every demonstration is an anchor.

A window at anchor frame t holds 9 video frames at t + 0, 4, ..., 32 (encoded once to latents, see
scripts/precompute_latents.py), the 32 actions t .. t + 31, the state at t and the instruction. Offsets past the end
of the episode repeat its last frame/action and are masked out of the loss. On LIBERO the padded actions keep only
the gripper (the six delta-pose dimensions are zero), as in FastWAM.

`scripts/prepare_data.py` writes a prepared directory with the window arrays (windows.npz), the prompts
(prompts.json), the per-episode video sources (episodes.json), the normalization statistics (dataset_stats.json)
and info.json; `Windows` reads it.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from .normalizer import MODES, load_normalizers
from .text import TextCache

ACTION_HORIZON = 32
VIDEO_OFFSETS = tuple(range(0, 33, 4))
LATENT_SHAPES = {'libero': (16, 3, 28, 56), 'robotwin': (16, 3, 48, 40), 'robocasa_gr1': (16, 3, 32, 32)}
DIMS = {'libero': (7, 8), 'robotwin': (14, 14), 'robocasa_gr1': (29, 29)}   # (action, proprio)


class Windows:
    def __init__(self, prepared_dir, text_cache_dir=None, text_memory=4096):
        root = Path(prepared_dir)
        self.info = json.loads((root / 'info.json').read_text())
        self.benchmark = self.info['benchmark']
        arrays = np.load(root / 'windows.npz')
        self.actions = torch.from_numpy(arrays['actions'])
        self.states = torch.from_numpy(arrays['states'])
        self.prompt_id = arrays['prompt_id']
        self.episode_starts = arrays['episode_starts']
        self.episode_lengths = arrays['episode_lengths']
        self.size = len(self.actions)
        self.episode_of = np.repeat(np.arange(len(self.episode_starts)), self.episode_lengths)
        self.ep_start = self.episode_starts[self.episode_of]
        self.ep_end = self.ep_start + self.episode_lengths[self.episode_of]
        if self.ep_end[-1] != self.size or self.info['windows'] != self.size:
            raise ValueError(f'{root}: inconsistent window index')
        self.prompts = json.loads((root / 'prompts.json').read_text())
        self.stats = json.loads((root / 'dataset_stats.json').read_text())
        self.normalizer = load_normalizers(self.stats, MODES[self.benchmark])
        self.latent_shape = LATENT_SHAPES[self.benchmark]
        self.action_dim, self.proprio_dim = DIMS[self.benchmark]
        self.text = TextCache(text_cache_dir or root / 'text', capacity=text_memory)

    def __len__(self):
        return self.size

    def episodes(self):
        return json.loads((Path(self.info['prepared_dir']) / 'episodes.json').read_text())

    def prompt(self, index):
        return self.prompts[int(self.prompt_id[index])]

    def fields(self, index: int) -> dict:
        start, end = int(self.ep_start[index]), int(self.ep_end[index])
        a_idx = index + np.arange(ACTION_HORIZON)
        action_is_pad = torch.from_numpy(a_idx >= end)
        action = self.actions[np.clip(a_idx, start, end - 1)].clone()
        if self.benchmark == 'libero':
            action[action_is_pad, :6] = 0
        context, mask = self.text(self.prompt(index))
        return dict(action=self.normalizer['action'].forward(action),
                    proprio=self.normalizer['state'].forward(self.states[index]),
                    action_is_pad=action_is_pad,
                    image_is_pad=torch.from_numpy(index + np.asarray(VIDEO_OFFSETS) >= end),
                    context=context, context_mask=mask)

    def batch(self, indices) -> dict:
        rows = [self.fields(int(i)) for i in indices]
        return {k: torch.stack([r[k] for r in rows]) for k in rows[0]}


def write_prepared(out_dir, benchmark, actions, states, prompt_id, prompts, episode_starts, episode_lengths,
                   episodes, stats, extra_arrays=None):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / 'windows.npz', actions=np.asarray(actions, np.float32), states=np.asarray(states, np.float32),
             prompt_id=np.asarray(prompt_id, np.int32), episode_starts=np.asarray(episode_starts, np.int64),
             episode_lengths=np.asarray(episode_lengths, np.int64), **(extra_arrays or {}))
    (out / 'prompts.json').write_text(json.dumps(prompts, indent=1))
    (out / 'episodes.json').write_text(json.dumps(episodes))
    (out / 'dataset_stats.json').write_text(json.dumps(stats, indent=2))
    (out / 'info.json').write_text(json.dumps(dict(
        benchmark=benchmark, prepared_dir=str(out.resolve()), windows=int(len(actions)), episodes=len(episodes),
        prompts=len(prompts), latent_shape=list(LATENT_SHAPES[benchmark]), action_dim=DIMS[benchmark][0],
        proprio_dim=DIMS[benchmark][1], normalization=MODES[benchmark]), indent=2))
