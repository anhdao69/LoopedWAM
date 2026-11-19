#!/usr/bin/env python3
"""Encode every training instruction once with the Wan UMT5-XXL text encoder (128 tokens).

    python scripts/precompute_text_embeds.py --data data/prepared/libero --wan checkpoints/Wan2.1-T2V-1.3B

Writes <data>/text/<sha256(prompt)>.pt = {'context': bf16 [128, 4096], 'mask': bool [128]} for each prompt in
<data>/prompts.json. Evaluation encodes new instructions with the same encoder.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from loopedwam.data.text import encode_prompts, load_text_encoder, prompt_path
from loopedwam.utils import atomic_save


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--data', required=True, help='prepared data directory (scripts/prepare_data.py)')
    p.add_argument('--wan', required=True, help='Wan2.1-T2V-1.3B directory (models_t5_umt5-xxl-enc-bf16.pth, google/)')
    p.add_argument('--out', help='output directory (default: <data>/text)')
    p.add_argument('--batch', type=int, default=16)
    a = p.parse_args()
    out = Path(a.out or Path(a.data) / 'text')
    prompts = [q for q in json.loads((Path(a.data) / 'prompts.json').read_text()) if not prompt_path(out, q).exists()]
    encoder, tokenizer = load_text_encoder(a.wan, 'cuda')
    for lo in range(0, len(prompts), a.batch):
        batch = prompts[lo:lo + a.batch]
        for prompt, (context, mask) in zip(batch, encode_prompts(encoder, tokenizer, batch)):
            atomic_save(dict(context=context, mask=mask), prompt_path(out, prompt))
        print(json.dumps(dict(encoded=min(lo + a.batch, len(prompts)), of=len(prompts))), flush=True)


if __name__ == '__main__':
    main()
