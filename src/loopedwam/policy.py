"""Evaluation policy: raw state and instruction + a preprocessed image -> denormalized 32-action chunk.

Loads a LoopedWAM checkpoint (4 action loops, any number of denoising steps) or a LoopedWAM-E checkpoint (any of the
trained settings K_a in {1, 2, 4} x N in {2, 5, 10}). Every query draws its initial noise from a CPU generator seeded
with `seed`, so a query is a deterministic function of the observation.
"""
from __future__ import annotations

import numpy as np
import torch

from loopedwam.data.normalizer import MODES, load_normalizers
from loopedwam.data.text import DEFAULT_PROMPT, PromptEncoder
from loopedwam.models.elastic import SCHEDULES, load_elastic
from loopedwam.models.loopedwam import ACTION_HORIZON, load_checkpoint


class Policy:
    def __init__(self, checkpoint, dataset_stats, vae, wan, *, benchmark, action_loops=4, steps=10, seed=42,
                 device='cuda', text_cache=None):
        header = torch.load(checkpoint, map_location='cpu', weights_only=False, mmap=True)
        self.elastic = 'student_action' in header
        del header
        if self.elastic:
            if f'flat_k{action_loops}_n{steps}' not in SCHEDULES:
                raise ValueError('LoopedWAM-E runs K_a in {1, 2, 4} action loops with N in {2, 5, 10} steps')
            self.model, _ = load_elastic(checkpoint, vae, device=device)
        else:
            if action_loops != 4:
                raise ValueError('LoopedWAM runs 4 action loops; use a LoopedWAM-E checkpoint for other depths')
            self.model, _ = load_checkpoint(checkpoint, vae, device=device)
        self.model.eval().requires_grad_(False)
        self.action_dim = (self.model.parent if self.elastic else self.model).action_dim
        self.normalizer = load_normalizers(dataset_stats, MODES[benchmark])
        self.text = PromptEncoder(wan, device, text_cache)
        self.action_loops, self.steps, self.seed = int(action_loops), int(steps), int(seed)

    def __call__(self, image, state, instruction):
        """image: [3, H, W] in [-1, 1] (benchmark layout); state: raw proprioception [D]; instruction: task text.
        Returns the denormalized action chunk, float32 [32, A]."""
        proprio = self.normalizer['state'].forward(torch.as_tensor(np.asarray(state, dtype=np.float32)))
        context, mask = self.text(DEFAULT_PROMPT.format(task=instruction))
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
            if self.elastic:
                action = self.model.infer_action(image, proprio, context, mask, seed=self.seed,
                                                 action_loops=self.action_loops, steps=self.steps)
            else:
                action = self.model.infer_action(image, proprio, context, mask, seed=self.seed,
                                                 num_inference_steps=self.steps)
        if action.shape != (ACTION_HORIZON, self.action_dim) or not torch.isfinite(action).all():
            raise ValueError(f'Invalid action chunk {tuple(action.shape)}')
        return self.normalizer['action'].backward(action[None])[0].numpy()
