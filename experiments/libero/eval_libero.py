#!/usr/bin/env python3
"""LIBERO evaluation of a LoopedWAM or LoopedWAM-E checkpoint: 4 suites x 10 tasks x 50 initial states.

    python experiments/libero/eval_libero.py --checkpoint checkpoints/LoopedWAM/libero/LoopedWAM/policy.pt \
        --gpus 0,1,2,3 --out evaluate_results/libero/LoopedWAM

    # LoopedWAM-E at K_a = 2 action loops and N = 5 denoising steps
    python experiments/libero/eval_libero.py --checkpoint checkpoints/LoopedWAM/libero/LoopedWAM-E/policy.pt \
        --action-loops 2 --steps 5 --gpus 0,1,2,3 --out evaluate_results/libero/LoopedWAM-E_k2_n5

Protocol (FastWAM): one environment per task, seeded with --seed and reused for the task's 50 official initial
states in order; 30 no-op settling steps, then policy control for up to 400 steps (Spatial, Object, Goal) or 700
(Long); each query predicts 32 actions and the first 10 are executed; the gripper command is sign(1 - 2g) of the
denormalized gripper action g (0 closed, 1 open); success is the environment's done flag.
Inputs: agent-view and wrist images (rotated 180 degrees), each resized to 224x224 and placed side by side; the 8-D
state is end-effector position, axis-angle orientation and the two gripper joints.

One worker per GPU loads the policy once and takes tasks from a shared queue. Each finished task is written to
<out>/<suite>/task<k>.json and skipped when the command is run again; <out>/summary.json holds the success rates.
"""
from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
from pathlib import Path

import numpy as np

SUITES = ('libero_spatial', 'libero_object', 'libero_goal', 'libero_10')
MAX_STEPS = {'libero_spatial': 400, 'libero_object': 400, 'libero_goal': 400, 'libero_10': 700}
SETTLING_STEPS = 30
EXECUTED_ACTIONS = 10
NOOP = [0., 0., 0., 0., 0., 0., -1.]
RESOLUTION = 256


def quat2axisangle(quat):
    """robosuite's quaternion (x, y, z, w) -> axis-angle."""
    quat = np.array(quat, dtype=np.float64, copy=True)
    quat[3] = np.clip(quat[3], -1.0, 1.0)
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def camera_images(obs):
    """Agent-view and wrist images rotated by 180 degrees, as in the training data."""
    return (np.ascontiguousarray(obs['agentview_image'][::-1, ::-1]),
            np.ascontiguousarray(obs['robot0_eye_in_hand_image'][::-1, ::-1]))


def observation(obs):
    """-> image [3, 224, 448] in [-1, 1], raw 8-D state."""
    import torch
    from torchvision.transforms.functional import resize
    views = [resize(torch.from_numpy(view).permute(2, 0, 1).float() / 255, [224, 224], antialias=True)
             for view in camera_images(obs)]
    image = (torch.cat(views, dim=-1) * 2 - 1).contiguous()
    state = np.concatenate([obs['robot0_eef_pos'], quat2axisangle(obs['robot0_eef_quat']), obs['robot0_gripper_qpos']])
    return image, state


def to_libero_actions(chunk):
    actions = np.array(chunk, dtype=np.float32, copy=True)
    actions[:, -1] = np.sign(-(2 * actions[:, -1] - 1))
    return actions


def run_task(policy, suite_name, task_id, args):
    import random

    import torch
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    task = benchmark.get_benchmark_dict()[suite_name]().get_task(task_id)
    initial_states = torch.load(Path(get_libero_path('init_states')) / task.problem_folder / task.init_states_file,
                                map_location='cpu', weights_only=False)
    env = OffScreenRenderEnv(bddl_file_name=str(Path(get_libero_path('bddl_files')) / task.problem_folder /
                                                task.bddl_file),
                             camera_heights=RESOLUTION, camera_widths=RESOLUTION)
    env.seed(args.seed)
    episodes = []
    try:
        for episode in range(args.trials):
            env.reset()
            obs = env.set_init_state(initial_states[episode])
            for _ in range(SETTLING_STEPS):
                obs, _, done, _ = env.step(NOOP)
            frames, pending, steps, done = [], [], 0, False
            while steps < MAX_STEPS[suite_name]:
                if not pending:
                    image, state = observation(obs)
                    pending = to_libero_actions(policy(image, state, task.language))[:EXECUTED_ACTIONS].tolist()
                if args.videos:
                    frames.append(np.concatenate(camera_images(obs), axis=1))
                obs, _, done, _ = env.step(pending.pop(0))
                steps += 1
                if done:
                    break
            episodes.append(dict(episode=episode, success=bool(done), steps=steps))
            if args.videos:
                import imageio
                path = Path(args.out) / 'videos' / suite_name / f'task{task_id}_episode{episode}_success{int(done)}.mp4'
                path.parent.mkdir(parents=True, exist_ok=True)
                imageio.mimwrite(path, frames, fps=20)
    finally:
        env.close()
    return dict(suite=suite_name, task=task_id, instruction=task.language, seed=args.seed,
                successes=sum(e['success'] for e in episodes), trials=len(episodes), episodes=episodes)


def worker(gpu, args, todo):
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu
    os.environ['MUJOCO_EGL_DEVICE_ID'] = gpu
    from loopedwam.policy import Policy
    policy = Policy(args.checkpoint, args.dataset_stats, args.vae, args.wan, benchmark='libero',
                    action_loops=args.action_loops, steps=args.steps, seed=args.seed, text_cache=args.text_cache)
    while (item := todo.get()) is not None:
        suite_name, task_id = item
        result = run_task(policy, suite_name, task_id, args)
        path = Path(args.out) / suite_name / f'task{task_id}.json'
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=1))
        print(json.dumps(dict(gpu=gpu, suite=suite_name, task=task_id, successes=result['successes'],
                              trials=result['trials'])), flush=True)


def summarize(args):
    summary = dict(checkpoint=args.checkpoint, action_loops=args.action_loops, steps=args.steps, seed=args.seed,
                   suites={})
    for suite_name in args.suites:
        results = [json.loads((Path(args.out) / suite_name / f'task{t}.json').read_text()) for t in range(10)]
        successes, trials = sum(r['successes'] for r in results), sum(r['trials'] for r in results)
        summary['suites'][suite_name] = dict(success_rate=successes / trials, successes=successes, trials=trials,
                                             per_task=[r['successes'] / r['trials'] for r in results])
    summary['average'] = float(np.mean([s['success_rate'] for s in summary['suites'].values()]))
    (Path(args.out) / 'summary.json').write_text(json.dumps(summary, indent=1))
    print(json.dumps({**{k: round(100 * v['success_rate'], 2) for k, v in summary['suites'].items()},
                      'average': round(100 * summary['average'], 2)}), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--dataset-stats', help='default: dataset_stats.json next to the checkpoint')
    p.add_argument('--vae', default='checkpoints/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth')
    p.add_argument('--wan', default='checkpoints/Wan2.1-T2V-1.3B', help='T5 encoder and tokenizer directory')
    p.add_argument('--text-cache', help='precomputed instruction embeddings (scripts/precompute_text_embeds.py)')
    p.add_argument('--action-loops', type=int, default=4, help='K_a (LoopedWAM-E: 1, 2 or 4)')
    p.add_argument('--steps', type=int, default=10, help='denoising steps N (LoopedWAM-E: 2, 5 or 10)')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--suites', default=','.join(SUITES))
    p.add_argument('--trials', type=int, default=50)
    p.add_argument('--gpus', default='0')
    p.add_argument('--videos', action='store_true')
    p.add_argument('--out', required=True)
    args = p.parse_args()
    args.suites = args.suites.split(',')
    args.dataset_stats = args.dataset_stats or str(Path(args.checkpoint).with_name('dataset_stats.json'))
    os.environ.setdefault('MUJOCO_GL', 'egl')
    os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')
    from libero.libero import benchmark  # noqa: F401  (on first use LIBERO asks here where to keep its config)
    todo = [(s, t) for s in args.suites for t in range(10) if not (Path(args.out) / s / f'task{t}.json').exists()]
    if todo:
        ctx = mp.get_context('spawn')
        queue = ctx.Queue()
        gpus = args.gpus.split(',')
        for item in todo + [None] * len(gpus):
            queue.put(item)
        workers = [ctx.Process(target=worker, args=(gpu, args, queue)) for gpu in gpus]
        for w in workers:
            w.start()
        for w in workers:
            w.join()
        if any(w.exitcode for w in workers):
            raise SystemExit('a worker failed; finished tasks are kept, run the same command again to resume')
    summarize(args)


if __name__ == '__main__':
    main()
