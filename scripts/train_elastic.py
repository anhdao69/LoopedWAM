#!/usr/bin/env python3
"""Train LoopedWAM-E from a trained LoopedWAM teacher (configs/<benchmark>/elastic.yaml).

    torchrun --standalone --nproc_per_node=4 scripts/train_elastic.py --config configs/libero/elastic.yaml

Every optimizer update trains one of the nine depth/step settings (K_a in {1, 2, 4} x N in {2, 5, 10}); each block of
nine consecutive updates visits all nine in a seeded random order. The student (a copy of the teacher's action
expert plus the step condition) is distilled from the teacher's 10-step trajectory, with an endpoint loss and the
native flow-matching loss as an anchor. Deterministic CUDA algorithms; FP32 weights, BF16 autocast.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import time
from pathlib import Path

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import torch
import torch.distributed as dist
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP

from loopedwam.data.latents import LatentCache
from loopedwam.data.plans import elastic_batches, elastic_noise, updates_per_epoch
from loopedwam.data.windows import Windows
from loopedwam.models.elastic import SCHEDULES, LoopedWAME, balanced_schedule_sequence, elastic_payload
from loopedwam.models.loopedwam import load_checkpoint
from loopedwam.utils import atomic_save, lr_factor


def elastic_batch(windows, cache, indices):
    """First-frame latents and action/proprio/text fields; index -1 is a zero-weight padding sample."""
    real = [max(i, 0) for i in indices]
    batch = windows.batch(real)
    batch['proprio'] = batch['proprio'][:, None]
    batch['first_frame_latents'] = cache.read(real, frames=1)
    batch['sample_valid'] = torch.tensor([i >= 0 for i in indices])
    del batch['image_is_pad']
    return batch


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--config', required=True)
    p.add_argument('--resume', action='store_true', help='continue from <output>/latest.pt')
    a = p.parse_args()
    cfg = yaml.safe_load(Path(a.config).read_text())
    rank, world, local = (int(os.environ.get(k, d)) for k, d in (('RANK', 0), ('WORLD_SIZE', 1), ('LOCAL_RANK', 0)))
    torch.use_deterministic_algorithms(True)
    torch.cuda.set_device(local)
    device = torch.device('cuda', local)
    if world > 1:
        dist.init_process_group('nccl', device_id=device)
    seed = cfg['seed']
    torch.manual_seed(seed)
    out = Path(cfg['output'])
    if rank == 0:
        out.mkdir(parents=True, exist_ok=True)
        if not a.resume and (out / 'latest.pt').exists():
            raise SystemExit(f'{out} already holds a run; pass --resume to continue it')
    if cfg['global_batch'] % (world * cfg['microbatch']):
        raise SystemExit('global_batch must be a multiple of world size x microbatch')

    windows = Windows(cfg['data'], cfg.get('text'))
    cache = LatentCache(cfg.get('latents') or Path(cfg['data']) / 'latents', len(windows), windows.latent_shape)
    if not cache.complete():
        raise SystemExit('The latent cache is incomplete: run scripts/precompute_latents.py first')
    teacher, teacher_payload = load_checkpoint(cfg['teacher'], cfg['vae'], device=device)
    policy = LoopedWAME(teacher, end_prefix=cfg['end_prefix'], gripper_loss=cfg['gripper_loss'],
                        checkpoint_blocks=cfg['checkpoint_blocks'], seed=seed).to(device)
    params = [p for p in policy.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(policy.optimizer_groups(), betas=(0.9, 0.95), eps=1e-8, fused=True)
    epoch0, micro0, update = 0, 0, 0
    if a.resume:
        state = torch.load(out / 'latest.pt', map_location='cpu', weights_only=False)
        policy.student_action.load_state_dict(state['student_action'])
        policy.step_condition.load_state_dict(state['step_condition'])
        opt.load_state_dict(state['optimizer'])
        epoch0, micro0, update = state['epoch'], state['next_micro'], state['update']
        del state
    runner = DDP(policy, device_ids=[local], broadcast_buffers=False, gradient_as_bucket_view=True) if world > 1 \
        else policy

    size, epochs = len(windows), cfg['epochs']
    total = updates_per_epoch(size, cfg['global_batch']) * epochs
    warmup = math.ceil(0.03 * total)
    sequence = balanced_schedule_sequence(seed, total)
    accumulation = cfg['global_batch'] // (world * cfg['microbatch'])
    micros_per_epoch = math.ceil(size / (cfg['microbatch'] * world))
    normalizer = windows.normalizer['action']
    scale, offset = normalizer.scale.to(device), normalizer.offset.to(device)
    config = dict(cfg, schedules=list(SCHEDULES))

    def save(name, epoch, next_micro):
        if rank == 0:
            if name == 'latest.pt':
                atomic_save(dict(student_action=policy.student_action.state_dict(),
                                 step_condition=policy.step_condition.state_dict(), optimizer=opt.state_dict(),
                                 epoch=epoch, next_micro=next_micro, update=update), out / name)
            else:
                atomic_save(elastic_payload(policy, teacher_payload, windows.benchmark, update, config,
                                            dict(epoch=epoch, update=update)), out / name)
        if world > 1:
            dist.barrier()

    log = (out / 'metrics.jsonl').open('a') if rank == 0 else None
    policy.train()
    for epoch in range(epoch0, epochs):
        start = micro0 if epoch == epoch0 else 0
        logs, started = {}, time.time()
        for micro, indices in enumerate(elastic_batches(size, cfg['microbatch'], world, rank, seed, epoch, start),
                                        start):
            group = micro // accumulation
            count = min(cfg['global_batch'], size - group * cfg['global_batch'])
            boundary = (micro + 1) % accumulation == 0 or micro + 1 == micros_per_epoch
            if micro % accumulation == 0:
                name = sequence[update]
            sample = {k: v.to(device, non_blocking=True) for k, v in elastic_batch(windows, cache, indices).items()}
            sample.update(action_scale=scale, action_offset=offset)
            noise, anchor_noise, anchor_tau = elastic_noise(indices, epoch, seed, windows.action_dim)
            sync = runner.no_sync() if world > 1 and not boundary else contextlib.nullcontext()
            with sync:
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    loss, components = runner(sample, SCHEDULES[name], noise.to(device), anchor_noise.to(device),
                                              anchor_tau.to(device))
                (loss * world / count).backward()
            for key, value in components.items():
                logs[key] = logs.get(key, 0) + value
            if not boundary:
                continue
            grad = torch.nn.utils.clip_grad_norm_(params, 1.0, error_if_nonfinite=True)
            factor = lr_factor(update, total, warmup)
            for g in opt.param_groups:
                g['lr'] = g['base_lr'] * factor
            opt.step()
            opt.zero_grad(set_to_none=True)
            update += 1
            reduced = torch.stack(list(logs.values()))
            if world > 1:
                dist.all_reduce(reduced)
            if rank == 0:
                record = dict(update=update, of=total, epoch=epoch + 1, setting=name, grad_norm=float(grad),
                              seconds=round(time.time() - started, 3),
                              **{k: float(v) / count for k, v in zip(logs, reduced)})
                log.write(json.dumps(record) + '\n')
                log.flush()
                if update <= 10 or update % 100 == 0:
                    print(json.dumps(record), flush=True)
            logs, started = {}, time.time()
            epoch_done = micro + 1 == micros_per_epoch
            if update % cfg['checkpoint_every'] == 0 or epoch_done:
                save('latest.pt', epoch + int(epoch_done), 0 if epoch_done else micro + 1)
            if epoch_done and epoch + 1 in cfg['save_epochs']:
                save(f'epoch_{epoch + 1:02d}.pt', epoch + 1, 0)
    if world > 1:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
