#!/usr/bin/env python3
"""LoopedWAM / LoopedWAM-E policy server for RoboCasa-GR1 (starVLA websocket protocol, msgpack-numpy frames).

    PYTHONPATH=third_party/starVLA python experiments/robocasa_gr1/policy_server.py \
        --checkpoint checkpoints/LoopedWAM/robocasa_gr1/LoopedWAM/policy.pt --port 6500

Request payload: {'observation': {...}, 'n_action_steps': n} with
  video.ego_view                   uint8 (B, T, 256, 256, 3); the latest frame is used
  state.{left_arm, right_arm, left_hand, right_hand, waist}   (B, T, 7 | 7 | 6 | 6 | 3) joint positions
  annotation.human.coarse_action   B instructions
Response data: action.{left_arm, right_arm, left_hand, right_hand, waist} (B, n, D), absolute joint targets: the first
n of the 32 predicted actions.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import torch

from loopedwam.data.robocasa_gr1 import GR1_GROUPS as GROUPS
from loopedwam.policy import Policy

VIDEO_KEY = 'video.ego_view'
TEXT_KEY = 'annotation.human.coarse_action'


class GR1Policy:
    def __init__(self, policy):
        self.policy = policy

    def predict_action(self, observation, n_action_steps=12):
        n = int(n_action_steps)
        if not 1 <= n <= 32:
            raise ValueError('n_action_steps must be in [1, 32]')
        video = np.asarray(observation[VIDEO_KEY])
        video = video[:, None] if video.ndim == 4 else video
        batch = len(video)
        texts = observation[TEXT_KEY]
        texts = [texts] if isinstance(texts, str) else [str(t) for t in texts]
        texts = texts * batch if len(texts) == 1 else texts
        states = np.concatenate([np.asarray(observation[f'state.{g}'], np.float32).reshape(batch, -1, d)[:, -1]
                                 for g, d in GROUPS], axis=1)
        chunks = []
        for b in range(batch):
            frame = np.array(video[b, -1], dtype=np.uint8, copy=True)
            if frame.shape != (256, 256, 3):
                raise ValueError(f'Expected a 256x256x3 ego-view frame, got {frame.shape}')
            image = (torch.from_numpy(frame).permute(2, 0, 1).float() / 255 * 2 - 1).to(torch.bfloat16)
            chunks.append(self.policy(image, states[b], texts[b]))
        chunks = np.stack(chunks).astype(np.float32)
        out, lo = {}, 0
        for g, d in GROUPS:
            out[f'action.{g}'] = np.ascontiguousarray(chunks[:, :n, lo:lo + d])
            lo += d
        return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--dataset-stats', help='default: dataset_stats.json next to the checkpoint')
    p.add_argument('--vae', default='checkpoints/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth')
    p.add_argument('--wan', default='checkpoints/Wan2.1-T2V-1.3B', help='T5 encoder and tokenizer directory')
    p.add_argument('--action-loops', type=int, default=4, help='K_a (LoopedWAM-E: 1, 2 or 4)')
    p.add_argument('--steps', type=int, default=10, help='denoising steps N (LoopedWAM-E: 2, 5 or 10)')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--host', default='0.0.0.0')
    p.add_argument('--port', type=int, default=6500)
    p.add_argument('--threads', type=int, default=8)
    a = p.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    torch.set_num_threads(a.threads)
    from deployment.model_server.tools.websocket_policy_server import WebsocketPolicyServer
    policy = Policy(a.checkpoint, a.dataset_stats or str(Path(a.checkpoint).with_name('dataset_stats.json')),
                    a.vae, a.wan, benchmark='robocasa_gr1', action_loops=a.action_loops, steps=a.steps, seed=a.seed)
    metadata = dict(policy='LoopedWAM', checkpoint=str(Path(a.checkpoint).resolve()), action_loops=a.action_loops,
                    steps=a.steps, seed=a.seed)
    WebsocketPolicyServer(policy=GR1Policy(policy), host=a.host, port=a.port, metadata=metadata).serve_forever()


if __name__ == '__main__':
    main()
