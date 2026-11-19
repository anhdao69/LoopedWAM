#!/usr/bin/env python3
"""Train LoopedWAM: one stage of a benchmark recipe (configs/<benchmark>/stage*.yaml).

    torchrun --standalone --nproc_per_node=4 scripts/train.py --config configs/libero/stage1.yaml

Flow matching on precomputed clip latents (scripts/precompute_latents.py) and cached T5 instructions
(scripts/precompute_text_embeds.py). FP32 weights and AdamW moments, BF16 autocast, gradient clipping at 1.0, linear
warmup then cosine decay to 1% of the peak learning rate. Every window's noise is a fixed function of the seed and the
window's position in the data plan, so the data and noise of each update do not depend on the GPU count, and --resume
continues an interrupted run at the same position in the data plan and learning-rate schedule.

Stage 1 starts from Wan2.1-T2V-1.3B (`wan`); stage 2 continues from the stage-1 checkpoint (`init`), optionally with
its optimizer state (`resume_optimizer`, RoboTwin and RoboCasa-GR1).
"""
from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import queue
import threading
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP

from loopedwam.data.latents import LatentCache
from loopedwam.data.plans import TraversalPlan, random_batches, slot_noise, slot_seed, updates_per_epoch
from loopedwam.data.windows import Windows
from loopedwam.models.loopedwam import build_loopedwam, checkpoint_payload, load_checkpoint
from loopedwam.utils import atomic_save, lr_factor

class Prefetcher:
    """Background thread that builds the CPU batches of the coming updates."""

    def __init__(self, items, windows, cache):
        self.q, self.error = queue.Queue(maxsize=2), None

        def work():
            try:
                for meta, micros in items:
                    built = []
                    for indices, keys, weight in micros:
                        batch = windows.batch(indices)
                        batch['latents_bits'] = cache.read_bits(indices)
                        built.append(({k: v.pin_memory() for k, v in batch.items()}, keys, weight))
                    self.q.put((meta, built))
            except BaseException as error:
                self.error = error
            self.q.put(None)

        threading.Thread(target=work, daemon=True).start()

    def __iter__(self):
        while (item := self.q.get()) is not None:
            yield item
        if self.error is not None:
            raise self.error


def random_plan(cfg, size, world, rank, start):
    """Updates of the random plan from cursor `start` = (epoch, batch, update)."""
    epoch0, batch0, update = start
    for epoch in range(epoch0, cfg['end_epoch']):
        for batch, (n, ids) in enumerate(random_batches(size, cfg['global_batch'], world, rank, cfg['seed'], epoch,
                                                        batch0 if epoch == epoch0 else 0),
                                         batch0 if epoch == epoch0 else 0):
            micros = []
            for lo in range(0, math.ceil(n / world), cfg['microbatch']):
                chosen = ids[lo:lo + cfg['microbatch']]
                keys = [(epoch, i, 0) for i in chosen] if chosen else [None]
                micros.append((chosen or [0], keys, world / n if chosen else 0.))
            last = batch + 1 == updates_per_epoch(size, cfg['global_batch'])
            yield dict(update=update, epoch=epoch, epoch_done=last, next=(epoch + int(last), 0 if last else batch + 1),
                       windows=n), micros
            update += 1


def traversal_plan(cfg, plan, world, rank, update):
    """Updates of the traversal plan; rank r owns streams r, r + world, ... (`streams_per_micro` per microstep)."""
    mine = list(range(rank, cfg['streams'], world))
    while not plan.exhausted:
        slots = plan.next_update()
        state = plan.state_dict()
        n = sum(x is not None for row in slots for x in row)
        micros = []
        for lo in range(0, len(mine), cfg['streams_per_micro']):
            streams = mine[lo:lo + cfg['streams_per_micro']]
            entries = [(slots[k][w], (update, k, w)) for w in range(cfg['windows_per_stream']) for k in streams
                       if slots[k][w] is not None]
            micros.append(([e[0] for e in entries], [e[1] for e in entries], world / n) if entries
                          else ([0], [None], 0.))
        yield dict(update=update, plan_state=state, windows=n, epoch_done=False), micros
        update += 1


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--config', required=True)
    p.add_argument('--resume', action='store_true', help='continue from <output>/latest.pt')
    a = p.parse_args()
    cfg = yaml.safe_load(Path(a.config).read_text())
    rank, world, local = (int(os.environ.get(k, d)) for k, d in (('RANK', 0), ('WORLD_SIZE', 1), ('LOCAL_RANK', 0)))
    torch.cuda.set_device(local)
    device = torch.device('cuda', local)
    if world > 1:
        dist.init_process_group('nccl', device_id=device)
    torch.manual_seed(cfg['seed'])
    out = Path(cfg['output'])
    if rank == 0:
        out.mkdir(parents=True, exist_ok=True)
        if not a.resume and (out / 'latest.pt').exists():
            raise SystemExit(f'{out} already holds a run; pass --resume to continue it')

    windows = Windows(cfg['data'], cfg.get('text'))
    cache = LatentCache(cfg.get('latents') or Path(cfg['data']) / 'latents', len(windows), windows.latent_shape)
    if not cache.complete():
        raise SystemExit('The latent cache is incomplete: run scripts/precompute_latents.py first')
    benchmark = windows.benchmark

    # Model and optimizer.
    resume = torch.load(out / 'latest.pt', map_location='cpu', weights_only=False, mmap=True) if a.resume else None
    if resume is not None:
        model, _ = load_checkpoint(out / 'latest.pt', cfg['vae'], device=device,
                                   checkpoint_blocks=cfg['checkpoint_blocks'])
    elif cfg.get('init'):
        model, init = load_checkpoint(cfg['init'], cfg['vae'], device=device, checkpoint_blocks=cfg['checkpoint_blocks'])
    else:
        model = build_loopedwam(windows.action_dim, windows.proprio_dim, cfg['vae'], device=device, wan_dir=cfg['wan'],
                                seed=cfg['seed'], checkpoint_blocks=cfg['checkpoint_blocks'])
    model.train()
    if cfg.get('compile'):
        torch._dynamo.config.cache_size_limit = 256
        torch._dynamo.config.accumulated_cache_size_limit = 8192
        torch._inductor.config.comprehensive_padding = False
        model.mot._joint_block = torch.compile(model.mot._joint_block)
    params = model.policy_parameters()
    opt = torch.optim.AdamW(params, lr=cfg['lr'], betas=(0.9, 0.95), eps=1e-8, weight_decay=cfg['weight_decay'],
                            fused=True)
    if resume is not None:
        opt.load_state_dict(resume['optimizer'])
    elif cfg.get('resume_optimizer'):
        opt.load_state_dict(init['optimizer'])
    runner = DDP(model, device_ids=[local], broadcast_buffers=False, gradient_as_bucket_view=True) if world > 1 \
        else model

    # Data plan and learning-rate schedule (counted in this stage's updates).
    state = resume['training_state'] if resume is not None else None
    if cfg['plan'] == 'random':
        per_epoch = updates_per_epoch(len(windows), cfg['global_batch'])
        total = per_epoch * (cfg['end_epoch'] - cfg['start_epoch'])
        cursor = (*state['next'], state['update']) if state else (cfg['start_epoch'], 0, 0)
        updates = random_plan(cfg, len(windows), world, rank, cursor)
    else:
        plan = TraversalPlan(windows.episode_starts, windows.episode_lengths, cfg['epochs'], cfg['seed'],
                             cfg['streams'], cfg['windows_per_stream'], cfg['stride'])
        total = plan.total_updates()
        if state:
            plan.load_state_dict(state['plan_state'])
        updates = traversal_plan(cfg, plan, world, rank, state['update'] if state else 0)
    warmup = int(total * cfg['warmup_fraction'])
    del resume

    def save(meta, name):
        if rank == 0:
            payload = checkpoint_payload(model, benchmark, meta['update'] + 1, cfg)
            if name == 'latest.pt':
                payload['optimizer'] = opt.state_dict()
                payload['training_state'] = dict(meta, update=meta['update'] + 1)
            atomic_save(payload, out / name)
        if world > 1:
            dist.barrier()

    log = (out / 'metrics.jsonl').open('a') if rank == 0 else None
    latent_shape = windows.latent_shape
    for meta, micros in Prefetcher(updates, windows, cache):
        started = time.time()
        for group in opt.param_groups:
            group['lr'] = cfg['lr'] * lr_factor(meta['update'], total, warmup)
        sums = torch.zeros(3, device=device)
        for i, (cpu, keys, weight) in enumerate(micros):
            batch = {k: v.to(device, non_blocking=True) for k, v in cpu.items() if k != 'latents_bits'}
            batch['latents'] = cpu['latents_bits'].to(device, non_blocking=True).view(torch.bfloat16).float()
            seeds = [slot_seed(cfg['seed'], *k) if k is not None else 0 for k in keys]
            noise = slot_noise(model, seeds, latent_shape, device)
            sync = runner.no_sync() if world > 1 and i < len(micros) - 1 else contextlib.nullcontext()
            with sync:
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    loss, logs = runner(batch, noise)
                (loss * weight).backward()
            if weight:
                sums += torch.stack([loss.detach(), logs['video'], logs['action']])
        grad = torch.nn.utils.clip_grad_norm_(params, 1.0, error_if_nonfinite=True)
        opt.step()
        opt.zero_grad(set_to_none=True)
        if world > 1:
            dist.all_reduce(sums)
        if rank == 0:
            n = meta['windows']
            record = dict(update=meta['update'] + 1, of=total, loss=float(sums[0]) / n, loss_video=float(sums[1]) / n,
                          loss_action=float(sums[2]) / n, grad_norm=float(grad), lr=opt.param_groups[0]['lr'],
                          seconds=round(time.time() - started, 3))
            log.write(json.dumps(record) + '\n')
            log.flush()
            if meta['update'] < 10 or (meta['update'] + 1) % 100 == 0:
                print(json.dumps(record), flush=True)
        done = meta['update'] + 1 == total
        if meta['epoch_done'] and meta['epoch'] + 1 in cfg.get('save_epochs', []):
            save(meta, f"epoch_{meta['epoch'] + 1:02d}.pt")
        if (meta['update'] + 1) % cfg['checkpoint_every'] == 0 or meta['epoch_done'] or done:
            save(meta, 'latest.pt')
        if done:
            save(meta, 'final.pt')
    if world > 1:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
