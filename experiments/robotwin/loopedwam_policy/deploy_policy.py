"""LoopedWAM / LoopedWAM-E policy for the RoboTwin 2.0 evaluation harness (script/eval_policy.py).

Inputs: the head camera resized to 320x256 above the two wrist cameras resized to 160x128 (PIL bilinear, as in
FastWAM), the 14-D joint state and the episode instruction. Each query predicts 32 absolute joint-position actions,
of which the first 24 are executed before the next observation is requested.
"""
from collections import deque

import numpy as np
import torch
from PIL import Image

from loopedwam.policy import Policy


def packed_image(observation):
    """[3, 384, 320] BF16 CUDA tensor in [-1, 1]."""
    cameras = observation['observation']

    def resize(name, size):
        rgb = Image.fromarray(cameras[name]['rgb'].astype(np.uint8), mode='RGB')
        return np.asarray(rgb.resize(size, resample=Image.BILINEAR), dtype=np.uint8)

    wrists = np.concatenate([resize('left_camera', (160, 128)), resize('right_camera', (160, 128))], axis=1)
    packed = np.concatenate([resize('head_camera', (320, 256)), wrists], axis=0)
    return torch.from_numpy(packed).permute(2, 0, 1).to(device='cuda', dtype=torch.bfloat16) * (2.0 / 255.0) - 1.0


class RoboTwinPolicy:
    def __init__(self, args):
        torch.set_num_threads(2)
        self.policy = Policy(args['ckpt_setting'], args['dataset_stats'], args['vae'], args['wan'],
                             benchmark='robotwin', action_loops=int(args['action_loops']), steps=int(args['steps']),
                             seed=int(args['seed']))
        self.replan_steps = int(args['replan_steps'])
        self.pending = deque()

    def reset(self):
        self.pending.clear()

    def should_request_observation(self):
        return not self.pending

    def step(self, task_env, observation):
        if not self.pending:
            state = np.asarray(observation['joint_action']['vector'], dtype=np.float32)
            chunk = self.policy(packed_image(observation), state, task_env.get_instruction())
            self.pending.extend(chunk[:self.replan_steps])
        task_env.take_action(self.pending.popleft(), action_type='qpos')


def get_model(args):
    return RoboTwinPolicy(args)


def eval(TASK_ENV, model, observation):
    model.step(TASK_ENV, observation)


def reset_model(model):
    model.reset()
