#!/usr/bin/env python3
"""RoboCasa-GR1 rollouts of one task against a policy server (runs in third_party/robocasa_env, cwd and PYTHONPATH
third_party/starVLA; experiments/robocasa_gr1/eval_robocasa_gr1.py starts it).

Uses starVLA's examples/simBenchmarks/Robocasa_tabletop/eval_files/simulation_env.py unchanged for the episode loop
(success at any step, n_action_steps environment steps per query, max_episode_steps per episode). starVLA builds its
environments unseeded; here environment i of a run with seed s is built with robosuite seed s * 1000 + i, so the
scenes are reproducible.
"""
from __future__ import annotations

import argparse
import json
from functools import partial
from pathlib import Path

import gymnasium as gym
import numpy as np

from deployment.model_server.tools.websocket_policy_client import WebsocketClientPolicy
from examples.simBenchmarks.Robocasa_tabletop.eval_files import simulation_env as sim

GROUPS = (('left_arm', 7), ('right_arm', 7), ('left_hand', 6), ('right_hand', 6), ('waist', 3))
TEXT_KEY = 'annotation.human.coarse_action'


class PolicyClient:
    """starVLA policy interface: step(observations) -> {'actions': {action.*: (B, n, D)}}."""

    def __init__(self, host, port, n_action_steps):
        self.client = WebsocketClientPolicy(host, port)
        self.n_action_steps = int(n_action_steps)
        self.calls = 0

    def step(self, observations):
        payload = {'video.ego_view': np.ascontiguousarray(observations['video.ego_view'], dtype=np.uint8)}
        for group, _ in GROUPS:
            payload[f'state.{group}'] = np.asarray(observations[f'state.{group}'], dtype=np.float64)
        text = observations[TEXT_KEY]
        text = text.tolist() if isinstance(text, np.ndarray) else list(text) if isinstance(text, (list, tuple)) else [text]
        payload[TEXT_KEY] = [str(t) for t in text]
        response = self.client.predict_action({'type': 'infer', 'request_id': f'q{self.calls}',
                                               'payload': {'observation': payload,
                                                           'n_action_steps': self.n_action_steps}})
        self.calls += 1
        if not response.get('ok', False):
            raise RuntimeError(f'policy server error: {response.get("error")}')
        batch = len(payload['video.ego_view'])
        actions = {}
        for group, dim in GROUPS:
            action = np.asarray(response['data'][f'action.{group}'], dtype=np.float32)
            if action.shape != (batch, self.n_action_steps, dim) or not np.isfinite(action).all():
                raise ValueError(f'action.{group}: invalid shape {action.shape}')
            actions[f'action.{group}'] = action
        return {'actions': actions}


def make_env(config, idx, seed):
    """starVLA's _create_single_env with an explicit robosuite seed."""
    env = gym.make(config.env_name, enable_render=True, seed=seed * 1000 + idx)
    if config.video.video_dir is not None:
        recorder = sim.VideoRecorder.create_h264(fps=config.video.fps, codec=config.video.codec,
                                                 input_pix_fmt=config.video.input_pix_fmt, crf=config.video.crf,
                                                 thread_type=config.video.thread_type,
                                                 thread_count=config.video.thread_count)
        env = sim.VideoRecordingWrapper(env, recorder, video_dir=Path(config.video.video_dir),
                                        steps_per_render=config.video.steps_per_render)
    return sim.MultiStepWrapper(env, video_delta_indices=config.multistep.video_delta_indices,
                                state_delta_indices=config.multistep.state_delta_indices,
                                n_action_steps=config.multistep.n_action_steps,
                                max_episode_steps=config.multistep.max_episode_steps)


class SeededSimulation(sim.SimulationInferenceEnv):
    def __init__(self, model, seed):
        super().__init__(model)
        self.seed = int(seed)

    def setup_environment(self, config):
        fns = [partial(make_env, config=config, idx=i, seed=self.seed) for i in range(config.n_envs)]
        if config.n_envs == 1:
            return gym.vector.SyncVectorEnv(fns)
        return gym.vector.AsyncVectorEnv(fns, shared_memory=False, context='spawn')


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--env-name', required=True, help='e.g. gr1_unified/PnPCupToDrawerClose_GR1ArmsAndWaistFourierHands_Env')
    p.add_argument('--host', default='127.0.0.1')
    p.add_argument('--port', type=int, required=True)
    p.add_argument('--episodes', type=int, default=50)
    p.add_argument('--n-envs', type=int, default=1)
    p.add_argument('--n-action-steps', type=int, default=12)
    p.add_argument('--max-episode-steps', type=int, default=720)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--video-dir')
    p.add_argument('--out', required=True, help='result JSON')
    a = p.parse_args()
    np.random.seed(a.seed)
    config = sim.SimulationConfig(env_name=a.env_name, n_episodes=a.episodes, n_envs=a.n_envs,
                                  video=sim.VideoConfig(video_dir=a.video_dir),
                                  multistep=sim.MultiStepConfig(n_action_steps=a.n_action_steps,
                                                                max_episode_steps=a.max_episode_steps))
    env_name, successes = SeededSimulation(PolicyClient(a.host, a.port, a.n_action_steps), a.seed).run_simulation(config)
    successes = [bool(s) for s in successes][:a.episodes]
    result = dict(env_name=env_name, seed=a.seed, episodes=a.episodes, n_action_steps=a.n_action_steps,
                  max_episode_steps=a.max_episode_steps, successes=sum(successes),
                  success_rate=float(np.mean(successes)), per_episode=successes)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(result, indent=1))
    print(json.dumps({k: result[k] for k in ('env_name', 'successes', 'episodes', 'success_rate')}), flush=True)


if __name__ == '__main__':
    main()
