#!/usr/bin/env python3
"""RoboCasa-GR1 evaluation of a LoopedWAM or LoopedWAM-E checkpoint: 24 tabletop tasks x 50 episodes.

    python experiments/robocasa_gr1/eval_robocasa_gr1.py \
        --checkpoint checkpoints/LoopedWAM/robocasa_gr1/LoopedWAM/policy.pt \
        --gpus 0,1,2,3 --out evaluate_results/robocasa_gr1/LoopedWAM

    # LoopedWAM-E at K_a = 4 action loops and N = 2 denoising steps
    python experiments/robocasa_gr1/eval_robocasa_gr1.py \
        --checkpoint checkpoints/LoopedWAM/robocasa_gr1/LoopedWAM-E/policy.pt \
        --action-loops 4 --steps 2 --gpus 0,1,2,3 --out evaluate_results/robocasa_gr1/LoopedWAM-E_k4_n2

Protocol (starVLA Robocasa_tabletop): one environment per task, 720 environment steps per episode, success at any
step, 12 of the 32 predicted actions executed per query, environment seed 0 and policy-noise seed 42.
Starts one policy server per GPU (policy_server.py, this environment) and runs up to --parallel simulator clients
(sim_client.py, third_party/robocasa_env, set up by setup_robocasa_gr1.sh) that render with EGL on the GPU of their
server. Each finished task is written to <out>/<task>.json and skipped when the command is run again;
<out>/summary.json holds the success rates.
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

from loopedwam.data.robocasa_gr1 import GR1_ENV_NAMES

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
STARVLA = ROOT / 'third_party' / 'starVLA'
SIM_PYTHON = ROOT / 'third_party' / 'robocasa_env' / 'bin' / 'python'


def port_open(port):
    with socket.socket() as s:
        s.settimeout(1)
        return s.connect_ex(('127.0.0.1', port)) == 0


def result_path(out, env_name):
    return out / f'{env_name.split("/")[-1]}.json'


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--dataset-stats', help='default: dataset_stats.json next to the checkpoint')
    p.add_argument('--vae', default='checkpoints/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth')
    p.add_argument('--wan', default='checkpoints/Wan2.1-T2V-1.3B', help='T5 encoder and tokenizer directory')
    p.add_argument('--action-loops', type=int, default=4, help='K_a (LoopedWAM-E: 1, 2 or 4)')
    p.add_argument('--steps', type=int, default=10, help='denoising steps N (LoopedWAM-E: 2, 5 or 10)')
    p.add_argument('--seed', type=int, default=42, help='policy noise seed')
    p.add_argument('--env-seed', type=int, default=0)
    p.add_argument('--episodes', type=int, default=50)
    p.add_argument('--n-action-steps', type=int, default=12)
    p.add_argument('--max-episode-steps', type=int, default=720)
    p.add_argument('--tasks', help='comma-separated environment names (default: all 24)')
    p.add_argument('--gpus', default='0')
    p.add_argument('--base-port', type=int, default=6500)
    p.add_argument('--parallel', type=int, default=24, help='simulator clients running at once')
    p.add_argument('--server-threads', type=int, default=8)
    p.add_argument('--sim-threads', type=int, default=2)
    p.add_argument('--videos', action='store_true')
    p.add_argument('--out', required=True)
    a = p.parse_args()
    out = Path(a.out).resolve()
    (out / 'logs').mkdir(parents=True, exist_ok=True)
    tasks = a.tasks.split(',') if a.tasks else list(GR1_ENV_NAMES)
    todo = [t for t in tasks if not result_path(out, t).exists()]
    gpus = a.gpus.split(',')
    ports = [a.base_port + i for i in range(len(gpus))]
    servers = []
    try:
        if todo:
            for gpu, port in zip(gpus, ports):
                if port_open(port):
                    raise SystemExit(f'port {port} is in use')
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, OMP_NUM_THREADS=str(a.server_threads),
                           PYTHONPATH=os.pathsep.join(filter(None, [str(STARVLA), os.environ.get('PYTHONPATH')])))
                cmd = [sys.executable, str(HERE / 'policy_server.py'), '--checkpoint', a.checkpoint,
                       '--vae', a.vae, '--wan', a.wan, '--action-loops', str(a.action_loops), '--steps', str(a.steps),
                       '--seed', str(a.seed), '--port', str(port), '--threads', str(a.server_threads)]
                cmd += ['--dataset-stats', a.dataset_stats] if a.dataset_stats else []
                servers.append(subprocess.Popen(cmd, env=env, stdout=open(out / 'logs' / f'server_{port}.log', 'w'),
                                                stderr=subprocess.STDOUT))
            deadline = time.time() + 900
            while not all(port_open(q) for q in ports):
                if any(s.poll() is not None for s in servers) or time.time() > deadline:
                    raise SystemExit(f'a policy server did not start; see {out}/logs/server_*.log')
                time.sleep(5)
        running, failed, k = [], [], 0
        while todo or running:
            while todo and len(running) < a.parallel:
                task = todo.pop(0)
                gpu, port = gpus[k % len(gpus)], ports[k % len(gpus)]
                k += 1
                threads = str(a.sim_threads)
                env = dict(os.environ, PYTHONPATH=str(STARVLA), NUMBA_DISABLE_JIT='1', OMP_NUM_THREADS=threads,
                           MKL_NUM_THREADS=threads, OPENBLAS_NUM_THREADS=threads, MUJOCO_GL='egl',
                           PYOPENGL_PLATFORM='egl', CUDA_VISIBLE_DEVICES=gpu, MUJOCO_EGL_DEVICE_ID=gpu)
                name = task.split('/')[-1]
                cmd = [str(SIM_PYTHON), str(HERE / 'sim_client.py'), '--env-name', task, '--port', str(port),
                       '--episodes', str(a.episodes), '--n-action-steps', str(a.n_action_steps),
                       '--max-episode-steps', str(a.max_episode_steps), '--seed', str(a.env_seed),
                       '--out', str(result_path(out, task))]
                cmd += ['--video-dir', str(out / 'videos' / name)] if a.videos else []
                log = open(out / 'logs' / f'{name}.log', 'w')
                running.append((subprocess.Popen(cmd, env=env, cwd=STARVLA, stdout=log, stderr=subprocess.STDOUT),
                                task, log))
            time.sleep(10)
            for item in list(running):
                process, task, log = item
                if process.poll() is None:
                    continue
                running.remove(item)
                log.close()
                if process.returncode or not result_path(out, task).exists():
                    failed.append(task)
                print(json.dumps(dict(task=task.split('/')[-1], exit=process.returncode, remaining=len(todo))),
                      flush=True)
            if any(s.poll() is not None for s in servers):
                raise SystemExit(f'a policy server stopped; see {out}/logs/server_*.log')
        if failed:
            raise SystemExit(f'failed: {failed}; see {out}/logs and run the same command again to resume')
    finally:
        for s in servers:
            s.terminate()
        for s in servers:
            try:
                s.wait(60)
            except subprocess.TimeoutExpired:
                s.kill()
    rows = {t.split('/')[-1]: json.loads(result_path(out, t).read_text()) for t in tasks}
    summary = dict(checkpoint=a.checkpoint, action_loops=a.action_loops, steps=a.steps, seed=a.seed,
                   env_seed=a.env_seed, n_action_steps=a.n_action_steps,
                   success_rate=sum(r['success_rate'] for r in rows.values()) / len(rows),
                   successes=sum(r['successes'] for r in rows.values()),
                   episodes=sum(r['episodes'] for r in rows.values()),
                   per_task={t: r['success_rate'] for t, r in rows.items()})
    (out / 'summary.json').write_text(json.dumps(summary, indent=1))
    print(json.dumps(dict(success_rate=round(100 * summary['success_rate'], 2), successes=summary['successes'],
                          episodes=summary['episodes'])), flush=True)


if __name__ == '__main__':
    main()
