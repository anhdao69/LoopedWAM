"""Looped two-expert transformer: pre(3) -> core(6) x 4 -> coda(3), sharing the core block weights.

The 12 physical video/action block pairs run on a virtual schedule of 3 + 6 * 4 + 3 = 30 layers. Every
virtual layer owns its own slot in the video KV cache used at inference.
"""
from __future__ import annotations

from functools import partial

import torch
from torch.utils.checkpoint import checkpoint

from .mot import MoT
from .wan_video_dit import flash_attention

PRE, CORE, CODA, LOOPS = 3, 6, 3, 4


def structured_mixed_attention(q_video, k_video, v_video, q_action, k_action, v_action,
                               observation_tokens: int, num_heads: int) -> torch.Tensor:
    """The canonical training mask written as mask-free attention calls: observed video reads observed video,
    future video reads all video, and actions read observed video plus actions."""
    n = observation_tokens
    parts = [flash_attention(q_video[:, :n], k_video[:, :n], v_video[:, :n], num_heads)]
    if q_video.shape[1] > n:
        parts.append(flash_attention(q_video[:, n:], k_video, v_video, num_heads))
    parts.append(flash_attention(q_action, torch.cat((k_video[:, :n], k_action), dim=1),
                                 torch.cat((v_video[:, :n], v_action), dim=1), num_heads))
    return torch.cat(parts, dim=1)


class LoopMoT(MoT):
    def __init__(self, mixtures, checkpoint_blocks: bool = False):
        if set(mixtures) != {'video', 'action'}:
            raise ValueError('LoopMoT requires the video and action experts.')
        super().__init__({name: mixtures[name] for name in ('video', 'action')})
        if self.num_layers != PRE + CORE + CODA:
            raise ValueError(f'LoopMoT requires {PRE + CORE + CODA} physical blocks per expert.')
        self.loops = LOOPS
        self.checkpoint_blocks = bool(checkpoint_blocks)
        # Training only: number of clean observation tokens for structured attention (None = masked attention).
        self.observation_tokens = None

    def virtual_schedule(self):
        """((stage, [one-based loop,] block), physical index) for the 30 virtual layers."""
        return (tuple((('pre', j), j) for j in range(PRE))
                + tuple((('core', r, j), PRE + j) for r in range(1, LOOPS + 1) for j in range(CORE))
                + tuple((('coda', LOOPS, j), PRE + CORE + j) for j in range(CODA)))

    def _post(self, block, io, mixed, context, context_mask):
        return self._apply_expert_post_block_tensor(block, io[3], mixed, *io[4:8], context, context_mask)

    def _joint_block(self, index, video_tokens, action_tokens, video_freqs, action_freqs, video_t_mod,
                     action_t_mod, video_context, video_context_mask, action_context, action_context_mask,
                     attention_mask):
        video_block = self.mixtures['video'].blocks[index]
        action_block = self.mixtures['action'].blocks[index]
        video_io = self._build_expert_attention_io(self.mixtures['video'], video_block, video_tokens, video_freqs,
                                                   video_t_mod)
        action_io = self._build_expert_attention_io(self.mixtures['action'], action_block, action_tokens,
                                                    action_freqs, action_t_mod)
        if self.observation_tokens is None:
            mixed = self._mixed_attention(torch.cat((video_io[0], action_io[0]), dim=1),
                                          torch.cat((video_io[1], action_io[1]), dim=1),
                                          torch.cat((video_io[2], action_io[2]), dim=1), attention_mask)
        else:
            mixed = structured_mixed_attention(*video_io[:3], *action_io[:3],
                                               observation_tokens=self.observation_tokens, num_heads=self.num_heads)
        n = video_tokens.shape[1]
        return (self._post(video_block, video_io, mixed[:, :n], video_context, video_context_mask),
                self._post(action_block, action_io, mixed[:, n:], action_context, action_context_mask))

    def _run_pair(self, index, video_tokens, action_tokens, conditioning):
        fn = partial(self._joint_block, index)
        if self.checkpoint_blocks and self.training and torch.is_grad_enabled():
            return checkpoint(fn, video_tokens, action_tokens, *conditioning, use_reentrant=False)
        return fn(video_tokens, action_tokens, *conditioning)

    def forward_joint(self, video_tokens, action_tokens, video_freqs, action_freqs, video_t_mod, action_t_mod,
                      video_context, video_context_mask, action_context, action_context_mask, attention_mask):
        """Training forward of the joint video/action token sequence through all 30 virtual layers."""
        conditioning = (video_freqs, action_freqs, video_t_mod, action_t_mod, video_context, video_context_mask,
                        action_context, action_context_mask, attention_mask)
        state = (video_tokens, action_tokens)
        for _, index in self.virtual_schedule():
            state = self._run_pair(index, *state, conditioning)
        return state

    def prefill_video_cache_tensor(self, video_tokens, video_freqs, video_t_mod, video_context, video_context_mask,
                                   video_attention_mask):
        """Run the clean observation through the video expert once and keep every virtual layer's keys/values."""
        expert = self.mixtures['video']
        x = video_tokens
        keys, values = [], []
        for _, index in self.virtual_schedule():
            block = expert.blocks[index]
            io = self._build_expert_attention_io(expert, block, x, video_freqs, video_t_mod)
            mixed = self._mixed_attention(*io[:3], video_attention_mask)
            keys.append(io[1])
            values.append(io[2])
            x = self._post(block, io, mixed, video_context, video_context_mask)
        return keys, values

    def forward_action_with_video_cache_tensor(self, action_tokens, action_freqs, action_t_mod, action_context,
                                               action_context_mask, video_cache_k, video_cache_v,
                                               action_attention_mask):
        """One action-expert pass; virtual layer i attends to the cached video keys/values of layer i."""
        expert = self.mixtures['action']
        x = action_tokens
        for slot, (_, index) in enumerate(self.virtual_schedule()):
            block = expert.blocks[index]
            io = self._build_expert_attention_io(expert, block, x, action_freqs, action_t_mod)
            mixed = self._mixed_attention(io[0], torch.cat((video_cache_k[slot], io[1]), dim=1),
                                          torch.cat((video_cache_v[slot], io[2]), dim=1), action_attention_mask)
            x = self._post(block, io, mixed, action_context, action_context_mask)
        return x
