#!/usr/bin/env python3
"""RoboTwin 2.0 evaluation of a LoopedWAM or LoopedWAM-E checkpoint: 50 tasks x {clean, randomized} x 100 episodes.

    python experiments/robotwin/eval_robotwin.py --checkpoint checkpoints/LoopedWAM/robotwin50/LoopedWAM/policy.pt \
        --gpus 0,1,2,3 --out evaluate_results/robotwin/LoopedWAM

    # LoopedWAM-E at K_a = 1 action loop and N = 2 denoising steps
    python experiments/robotwin/eval_robotwin.py \
        --checkpoint checkpoints/LoopedWAM/robotwin50/LoopedWAM-E/policy.pt \
        --action-loops 1 --steps 2 --gpus 0,1,2,3 --out evaluate_results/robotwin/LoopedWAM-E_k1_n2

Runs the official RoboTwin harness (third_party/RoboTwin/script/eval_policy.py, set up by setup_robotwin.sh) with the
policy in loopedwam_policy/: seed 42, unseen instructions, 100 accepted episodes per task and setting, 32 predicted
and 24 executed actions per query. Each (task, setting) is one harness process on one GPU; finished cells are
skipped when the command is run again. <out>/summary.json holds the success rates.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
ROBOTWIN = ROOT / 'third_party' / 'RoboTwin'
SETTINGS = {'demo_clean': 'clean', 'demo_randomized': 'random'}


def all_tasks():
    return list(yaml.safe_load((ROBOTWIN / 'task_config' / '_eval_step_limit.yml').read_text()))


def read_result(out, task, setting):
    path = out / task / setting / f'_result_{SETTINGS[setting]}.txt'
    if not path.exists():
        return None
    return float([line for line in path.read_text().splitlines() if line.strip()][-1])


def command(args, out, task, setting):
    overrides = dict(task_name=task, task_config=setting, ckpt_setting=Path(args.checkpoint).resolve(),
                     seed=args.seed, eval_num_episodes=args.episodes, eval_output_dir=out / task / setting,
                     dataset_stats=Path(args.dataset_stats).resolve(), vae=Path(args.vae).resolve(),
                     wan=Path(args.wan).resolve(), action_loops=args.action_loops, steps=args.steps)
    return [sys.executable, '-u', 'script/eval_policy.py', '--config', 'policy/loopedwam_policy/deploy_policy.yml',
            '--overrides', *[x for k, v in overrides.items() for x in (f'--{k}', str(v))]]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--dataset-stats', help='default: dataset_stats.json next to the checkpoint')
    p.add_argument('--vae', default='checkpoints/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth')
    p.add_argument('--wan', default='checkpoints/Wan2.1-T2V-1.3B', help='T5 encoder and tokenizer directory')
    p.add_argument('--action-loops', type=int, default=4, help='K_a (LoopedWAM-E: 1, 2 or 4)')
    p.add_argument('--steps', type=int, default=10, help='denoising steps N (LoopedWAM-E: 2, 5 or 10)')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--episodes', type=int, default=100)
    p.add_argument('--tasks', help='comma-separated task names (default: all 50)')
    p.add_argument('--gpus', default='0')
    p.add_argument('--jobs-per-gpu', type=int, default=1)
    p.add_argument('--out', required=True)
    args = p.parse_args()
    args.dataset_stats = args.dataset_stats or str(Path(args.checkpoint).with_name('dataset_stats.json'))
    out = Path(args.out).resolve()
    tasks = args.tasks.split(',') if args.tasks else all_tasks()
    todo = [(t, s) for t in tasks for s in SETTINGS if read_result(out, t, s) is None]
    free = [gpu for gpu in args.gpus.split(',') for _ in range(args.jobs_per_gpu)]
    running, failed = [], []
    while todo or running:
        while todo and free:
            task, setting = todo.pop(0)
            gpu = free.pop(0)
            (out / 'logs').mkdir(parents=True, exist_ok=True)
            log = open(out / 'logs' / f'{task}_{setting}.log', 'w')
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, PYTHONUNBUFFERED='1')
            process = subprocess.Popen(command(args, out, task, setting), cwd=ROBOTWIN, env=env, stdout=log,
                                       stderr=subprocess.STDOUT)
            running.append((process, gpu, task, setting, log))
        time.sleep(5)
        for item in list(running):
            process, gpu, task, setting, log = item
            if process.poll() is None:
                continue
            running.remove(item)
            log.close()
            free.append(gpu)
            rate = read_result(out, task, setting)
            if process.returncode or rate is None:
                failed.append(f'{task}/{setting}')
            print(json.dumps(dict(task=task, setting=setting, success_rate=rate, exit=process.returncode)), flush=True)
    if failed:
        raise SystemExit(f'failed: {failed}; see {out}/logs and run the same command again to resume')
    rows = {t: {SETTINGS[s]: read_result(out, t, s) for s in SETTINGS} for t in tasks}
    clean = sum(r['clean'] for r in rows.values()) / len(rows)
    randomized = sum(r['random'] for r in rows.values()) / len(rows)
    summary = dict(checkpoint=args.checkpoint, action_loops=args.action_loops, steps=args.steps, seed=args.seed,
                   clean=clean, randomized=randomized, average=(clean + randomized) / 2, per_task=rows)
    (out / 'summary.json').write_text(json.dumps(summary, indent=1))
    print(json.dumps(dict(clean=round(100 * clean, 2), randomized=round(100 * randomized, 2),
                          average=round(100 * summary['average'], 2))), flush=True)


if __name__ == '__main__':
    main()
