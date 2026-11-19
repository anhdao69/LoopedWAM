#!/usr/bin/env python3
"""Encode every training clip (9 frames, stride 4) once with the frozen Wan2.1 VAE.

    torchrun --standalone --nproc_per_node=4 scripts/precompute_latents.py --data data/prepared/libero \
        --vae checkpoints/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth

Video layout per benchmark (FastWAM):
  LIBERO       agent view | wrist view, each resized to 224x224, side by side: 224x448 -> latents [16, 3, 28, 56]
  RoboTwin     head camera (256x320) above the two wrist cameras (128x160 each): 384x320 -> latents [16, 3, 48, 40]
  RoboCasa-GR1 ego view at its native 256x256 -> latents [16, 3, 32, 32]
Each clip is encoded independently (the VAE cache is reset per clip). The run is resumable: rows already written
are skipped.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import av
import numpy as np
import torch
import torch.distributed as dist
from torchvision.transforms.functional import resize

from loopedwam.data.latents import LatentCache
from loopedwam.data.windows import VIDEO_OFFSETS, Windows
from loopedwam.models.donors import load_wan21_vae

BATCH = {'libero': 16, 'robotwin': 16, 'robocasa_gr1': 32}


def decode_at(path, timestamps, tolerance=1e-4):
    """Frames closest to `timestamps` (seconds), float [T, 3, H, W] in [0, 1]: seek to the keyframe before the
    first timestamp and decode until the last one (LeRobot's PyAV decoding)."""
    with av.open(str(path), metadata_errors='ignore') as container:
        stream = container.streams.video[0]
        container.seek(int(round(max(min(timestamps), 0) / stream.time_base)), backward=True, any_frame=False,
                       stream=stream)
        frames, loaded = [], []
        for frame in container.decode(video=0):
            loaded.append(float(frame.pts * frame.time_base))
            frames.append(torch.as_tensor(frame.to_rgb().to_ndarray()).permute(2, 0, 1))
            if loaded[-1] >= max(timestamps):
                break
    query = torch.tensor(timestamps, dtype=torch.float32)
    distance, nearest = torch.cdist(query[:, None], torch.tensor(loaded, dtype=torch.float32)[:, None], p=1).min(1)
    if not (distance < tolerance).all():
        raise ValueError(f'{path}: no decoded frame within {tolerance}s of some requested timestamps')
    return torch.stack([frames[i] for i in nearest]).float() / 255


def decode_all(path, last_frame, threads=2):
    """uint8 [T, 3, H, W] for frames 0..last_frame, decoded sequentially."""
    frames = []
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = 'AUTO'
        stream.codec_context.thread_count = threads
        for frame in container.decode(stream):
            frames.append(frame.to_ndarray(format='rgb24'))
            if len(frames) > last_frame:
                break
    if len(frames) <= last_frame:
        raise ValueError(f'{path}: decoded {len(frames)} frames, expected at least {last_frame + 1}')
    return torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2)


def robotwin_frames(views):
    """Three uint8 [T, 3, H, W] views -> bf16 [T, 3, 384, 320] in [-1, 1]."""
    resized = [resize(v.float() / 255, [240, 320], antialias=True) for v in views]
    high = resize(resized[0], [256, 320], antialias=True)
    wrists = torch.cat([resize(v, [128, 160], antialias=True) for v in resized[1:]], -1)
    return (torch.cat([high, wrists], -2) * 2 - 1).to(torch.bfloat16)


class EpisodeFrames(torch.utils.data.Dataset):
    """Item: (episode, frames needed by its windows [F, 3, H, W], their frame ids)."""

    def __init__(self, windows, episodes, work):
        self.windows, self.episodes, self.work = windows, episodes, work

    def __len__(self):
        return len(self.work)

    def __getitem__(self, i):
        d = self.work[i]
        start, length = int(self.windows.episode_starts[d]), int(self.windows.episode_lengths[d])
        episode, benchmark = self.episodes[d], self.windows.benchmark
        if benchmark == 'libero':
            ts = self.windows.timestamps[start:start + length].tolist()
            views = []
            for source in episode['videos'].values():
                frames = decode_at(source['path'], [source['from_timestamp'] + t for t in ts])
                views.append(torch.stack([resize(f, [224, 224], antialias=True) for f in frames]))
            frames = torch.cat(views, -1)       # float [T, 3, 224, 448] in [0, 1]
        elif benchmark == 'robotwin':
            frames = robotwin_frames([decode_all(path, length - 1) for path in episode['videos'].values()])
        else:
            frames = (decode_all(episode['video'], length - 1).float() / 255 * 2 - 1).to(torch.bfloat16)
        return d, frames


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--data', required=True, help='prepared data directory (scripts/prepare_data.py)')
    p.add_argument('--vae', required=True, help='Wan2.1_VAE.pth')
    p.add_argument('--out', help='latent cache directory (default: <data>/latents)')
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--episodes', type=int, help='encode only the first N episodes (smoke test)')
    a = p.parse_args()
    rank, world, local = (int(os.environ.get(k, d)) for k, d in (('RANK', 0), ('WORLD_SIZE', 1), ('LOCAL_RANK', 0)))
    torch.cuda.set_device(local)
    if world > 1:
        dist.init_process_group('nccl', device_id=torch.device('cuda', local))
    windows = Windows(a.data)
    if windows.benchmark == 'libero':
        windows.timestamps = np.load(Path(a.data) / 'windows.npz')['timestamps']
    out = Path(a.out or Path(a.data) / 'latents')
    if rank == 0:
        LatentCache(out, len(windows), windows.latent_shape, writable=True)
    if world > 1:
        dist.barrier()
    cache = LatentCache(out, len(windows), windows.latent_shape, writable=True)
    vae = load_wan21_vae(a.vae, device=f'cuda:{local}', dtype=torch.bfloat16)
    episodes = windows.episodes()
    todo = [d for d in range(len(episodes))[:a.episodes] if not np.all(
        cache.valid[windows.episode_starts[d]:windows.episode_starts[d] + windows.episode_lengths[d]] == 1)]
    load, mine = [0] * world, []
    for d in sorted(todo, key=lambda d: -int(windows.episode_lengths[d])):   # balance windows across ranks
        r = int(np.argmin(load))
        load[r] += int(windows.episode_lengths[d])
        if r == rank:
            mine.append(d)
    loader = torch.utils.data.DataLoader(EpisodeFrames(windows, episodes, mine), batch_size=None,
                                         num_workers=a.workers, prefetch_factor=2 if a.workers else None)
    offsets, batch, done, started = torch.tensor(VIDEO_OFFSETS), BATCH[windows.benchmark], 0, time.time()
    for n, (d, frames) in enumerate(loader):
        start, length = int(windows.episode_starts[d]), int(windows.episode_lengths[d])
        frames = frames.cuda(non_blocking=True)
        for lo in range(0, length, batch):
            anchors = torch.arange(lo, min(length, lo + batch))
            index = torch.clamp(anchors[:, None] + offsets, max=length - 1).cuda()
            video = frames[index]
            if windows.benchmark == 'libero':
                video = video * 2 - 1
            video = video.permute(0, 2, 1, 3, 4).contiguous()            # [B, 3, 9, H, W]
            with torch.no_grad():
                latents = vae.model.encode(video.to(torch.bfloat16), vae.scale)
                vae.model.clear_cache()
            cache.write((start + anchors).numpy(), latents.float())
            done += len(anchors)
        if n % 20 == 0 or n == len(mine) - 1:
            print(json.dumps(dict(rank=rank, episodes=n + 1, of=len(mine), windows=done,
                                  windows_per_second=round(done / (time.time() - started), 1))), flush=True)
    if world > 1:
        dist.barrier()
    if rank == 0:
        print(json.dumps(dict(event='done', valid=int(np.count_nonzero(cache.valid == 1)), windows=len(windows))))
    if world > 1:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
