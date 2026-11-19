"""T5 instruction embeddings: the Wan UMT5-XXL encoder (128 tokens), cached per prompt."""
from __future__ import annotations

import hashlib
from collections import OrderedDict
from pathlib import Path

import torch

DEFAULT_PROMPT = "A video recorded from a robot's point of view executing the following instruction: {task}"
CONTEXT_LEN, TEXT_DIM = 128, 4096


def prompt_path(cache_dir, prompt: str) -> Path:
    return Path(cache_dir) / f'{hashlib.sha256(prompt.encode("utf-8")).hexdigest()}.pt'


def read_text(path):
    """(context bf16 [128, 4096] with padding rows zeroed, mask bool [128]) from one cache file."""
    payload = torch.load(path, map_location='cpu', weights_only=True)
    mask = payload['mask'].bool()
    context = payload['context']
    if payload.get('packed_context', False):
        full = torch.zeros(CONTEXT_LEN, TEXT_DIM, dtype=context.dtype)
        full[mask] = context
        context = full
    else:
        context = context.clone()
    if context.shape != (CONTEXT_LEN, TEXT_DIM) or mask.shape != (CONTEXT_LEN,) or not mask.any():
        raise ValueError(f'Invalid text embedding file: {path}')
    context[~mask] = 0
    return context, mask


class TextCache:
    """Prompt -> (context, mask), read from `cache_dir`; keeps the `capacity` most recently used prompts in memory."""

    def __init__(self, cache_dir, capacity=None):
        self.cache_dir, self.capacity = Path(cache_dir), capacity
        self.memo = OrderedDict()

    def __call__(self, prompt: str):
        if prompt in self.memo:
            self.memo.move_to_end(prompt)
        else:
            self.memo[prompt] = read_text(prompt_path(self.cache_dir, prompt))
            if self.capacity is not None and len(self.memo) > self.capacity:
                self.memo.popitem(last=False)
        return self.memo[prompt]


def load_text_encoder(wan_dir, device='cuda'):
    """The UMT5-XXL encoder (BF16) and tokenizer shipped with Wan2.1-T2V-1.3B."""
    from loopedwam.models.wan_video_text_encoder import HuggingfaceTokenizer, WanTextEncoder
    wan_dir = Path(wan_dir)
    encoder = WanTextEncoder()
    state = torch.load(wan_dir / 'models_t5_umt5-xxl-enc-bf16.pth', map_location='cpu', weights_only=True)
    encoder.load_state_dict(state, strict=True)
    encoder = encoder.to(device=device, dtype=torch.bfloat16).eval().requires_grad_(False)
    tokenizer = HuggingfaceTokenizer(name=str(wan_dir / 'google/umt5-xxl'), seq_len=CONTEXT_LEN, clean='whitespace')
    return encoder, tokenizer


@torch.no_grad()
def encode_prompts(encoder, tokenizer, prompts, device='cuda'):
    """[(context bf16 [128, 4096], mask bool [128])] for a batch of prompts."""
    ids, mask = tokenizer(list(prompts), return_mask=True, add_special_tokens=True)
    mask = mask.to(device=device, dtype=torch.bool)
    context = encoder(ids.to(device), mask)
    out = []
    for i in range(len(prompts)):
        c = context[i].detach().to(device='cpu', dtype=torch.bfloat16).contiguous()
        m = mask[i].detach().cpu().contiguous()
        c[~m] = 0
        out.append((c, m))
    return out


class PromptEncoder:
    """Prompt -> (context, mask) at evaluation time: read from `cache_dir` (scripts/precompute_text_embeds.py) when
    the prompt is there, otherwise encoded with the T5 encoder, which is loaded on first use."""

    def __init__(self, wan_dir, device='cuda', cache_dir=None):
        self.wan_dir, self.device, self.cache_dir = wan_dir, device, cache_dir
        self.encoder = self.tokenizer = None
        self.memo = {}

    def __call__(self, prompt: str):
        if prompt not in self.memo:
            path = prompt_path(self.cache_dir, prompt) if self.cache_dir else None
            if path is not None and path.exists():
                self.memo[prompt] = read_text(path)
            else:
                if self.encoder is None:
                    self.encoder, self.tokenizer = load_text_encoder(self.wan_dir, self.device)
                self.memo[prompt] = encode_prompts(self.encoder, self.tokenizer, [prompt], self.device)[0]
        return self.memo[prompt]
