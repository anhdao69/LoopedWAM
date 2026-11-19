#!/usr/bin/env python3
"""Index the training windows of one benchmark and write its normalization statistics (CPU only).

    python scripts/prepare_data.py --benchmark libero --root data/libero --out data/prepared/libero
    python scripts/prepare_data.py --benchmark robotwin --root data/robotwin2.0/robotwin2.0 \
        --stats data/robotwin2.0/dataset_stats.json --out data/prepared/robotwin
    python scripts/prepare_data.py --benchmark robocasa_gr1 --root data/robocasa_gr1 --out data/prepared/robocasa_gr1

Statistics: LIBERO global min/max over all frames of the four suites; RoboTwin the published z-score statistics
of the FastWAM RoboTwin dataset; RoboCasa-GR1 global min/max over all frames of the 24 tasks (29 controlled joints).
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from loopedwam.data.robocasa_gr1 import GR1_FOLDERS, GR1_SELECT
from loopedwam.data.text import DEFAULT_PROMPT
from loopedwam.data.windows import write_prepared

LIBERO_SUITES = ('libero_spatial', 'libero_object', 'libero_goal', 'libero_10')
LIBERO_CAMERAS = ('observation.images.image', 'observation.images.wrist_image')
ROBOTWIN_CAMERAS = ('observation.images.cam_high', 'observation.images.cam_left_wrist',
                    'observation.images.cam_right_wrist')
GR1_CAMERA = 'observation.images.ego_view'


class Prompts:
    def __init__(self):
        self.prompts, self.ids = [], {}

    def __call__(self, task: str) -> int:
        prompt = DEFAULT_PROMPT.format(task=task)
        if prompt not in self.ids:
            self.ids[prompt] = len(self.prompts)
            self.prompts.append(prompt)
        return self.ids[prompt]


def vectors(table, column, dtype=np.float32):
    return np.stack(table.column(column).to_numpy(zero_copy_only=False)).astype(dtype)


def check_frames(frames, length, where):
    if len(frames) != length or not np.array_equal(frames, np.arange(length)):
        raise ValueError(f'{where}: rows are not one contiguous, ordered episode')


def prepare_libero(root):
    """Four LeRobot v3.0 suites, concatenated in LIBERO_SUITES order, episodes in ascending order."""
    prompts, actions, states, timestamps, prompt_id, lengths, episodes = Prompts(), [], [], [], [], [], []
    mins, maxs = {'action': [], 'state': []}, {'action': [], 'state': []}
    for suite in LIBERO_SUITES:
        directory = Path(root) / f'{suite}_no_noops_lerobot'
        info = json.loads((directory / 'meta/info.json').read_text())
        if info['codebase_version'] != 'v3.0':
            raise ValueError(f'{directory}: expected a LeRobot v3.0 dataset')
        meta = sorted(pq.read_table(sorted(map(str, (directory / 'meta/episodes').glob('*/*.parquet')))).to_pylist(),
                      key=lambda row: int(row['episode_index']))
        table = pq.read_table(sorted(map(str, (directory / 'data').glob('*/*.parquet'))),
                              columns=['episode_index', 'frame_index', 'timestamp', 'action', 'observation.state'])
        ep, frame = table.column('episode_index').to_numpy(), table.column('frame_index').to_numpy()
        act = np.asarray(table.column('action').to_pylist(), dtype=np.float32)
        st = np.asarray(table.column('observation.state').to_pylist(), dtype=np.float32)
        ts = table.column('timestamp').to_numpy().astype(np.float32)
        if len(meta) != info['total_episodes'] or len(act) != info['total_frames']:
            raise ValueError(f'{directory}: episode metadata does not match info.json')
        for kind, values in (('action', act), ('state', st)):
            mins[kind].append(values.min(0).tolist())
            maxs[kind].append(values.max(0).tolist())
        for row in meta:
            e, length = int(row['episode_index']), int(row['length'])
            if len(row['tasks']) != 1:
                raise ValueError(f'{suite} episode {e}: expected exactly one task')
            rows = np.flatnonzero(ep == e)
            rows = rows[np.argsort(frame[rows], kind='stable')]
            check_frames(frame[rows], length, f'{suite} episode {e}')
            actions.append(act[rows]); states.append(st[rows]); timestamps.append(ts[rows])
            prompt_id.append(np.full(length, prompts(row['tasks'][0])))
            lengths.append(length)
            episodes.append(dict(suite=suite, episode_index=e, videos={
                key: dict(path=str(directory / info['video_path'].format(
                    video_key=key, chunk_index=int(row[f'videos/{key}/chunk_index']),
                    file_index=int(row[f'videos/{key}/file_index']))),
                    from_timestamp=float(row[f'videos/{key}/from_timestamp'])) for key in LIBERO_CAMERAS}))
    stats = dict(num_episodes=len(episodes), num_transition=int(sum(lengths)))
    for kind in ('action', 'state'):
        stats[kind] = {'default': dict(global_min=np.min(mins[kind], axis=0).tolist(),
                                       global_max=np.max(maxs[kind], axis=0).tolist())}
    return (np.concatenate(actions), np.concatenate(states), np.concatenate(prompt_id), prompts.prompts, lengths,
            episodes, stats, dict(timestamps=np.concatenate(timestamps)))


def prepare_robotwin(root, stats_path, threads=16):
    """The 50 RoboTwin 2.0 tasks x (50 clean + 500 randomized) demonstrations, episodes in ascending order."""
    root = Path(root)
    info = json.loads((root / 'meta/info.json').read_text())
    if info['codebase_version'] != 'v2.1':
        raise ValueError(f'{root}: expected a LeRobot v2.1 dataset')
    meta = sorted((json.loads(line) for line in (root / 'meta/episodes.jsonl').read_text().splitlines() if line),
                  key=lambda row: int(row['episode_index']))
    tasks = {int(row['task_index']): row['task'] for row in
             (json.loads(line) for line in (root / 'meta/tasks.jsonl').read_text().splitlines() if line)}
    chunk = int(info['chunks_size'])

    def read(row):
        e, length = int(row['episode_index']), int(row['length'])
        table = pq.read_table(root / info['data_path'].format(episode_chunk=e // chunk, episode_index=e),
                              columns=['action', 'observation.state', 'task_index', 'episode_index', 'frame_index'])
        if not (table.column('episode_index').to_numpy() == e).all():
            raise ValueError(f'Episode {e}: foreign rows')
        check_frames(table.column('frame_index').to_numpy(), length, f'episode {e}')
        return (vectors(table, 'action'), vectors(table, 'observation.state'),
                table.column('task_index').to_numpy().astype(np.int64))

    with ThreadPoolExecutor(threads) as pool:
        parts = list(pool.map(read, meta))
    prompts = Prompts()
    prompt_id = np.concatenate([[prompts(tasks[int(t)]) for t in p[2]] for p in parts])
    episodes = [dict(episode_index=int(row['episode_index']), videos={
        key: str(root / info['video_path'].format(episode_chunk=int(row['episode_index']) // chunk, video_key=key,
                                                  episode_index=int(row['episode_index'])))
        for key in ROBOTWIN_CAMERAS}) for row in meta]
    stats = json.loads(Path(stats_path).read_text())
    return (np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts]), prompt_id, prompts.prompts,
            [int(row['length']) for row in meta], episodes, stats, None)


def prepare_robocasa_gr1(root, threads=16):
    """The 24 RoboCasa-GR1 tabletop tasks x 1000 demonstrations (LeRobot v2.0), folder-major."""
    prompts, parts, lengths, episodes = Prompts(), [], [], []
    for folder in GR1_FOLDERS:
        directory = Path(root) / folder
        info = json.loads((directory / 'meta/info.json').read_text())
        if not str(info['codebase_version']).startswith('v2.') or float(info['fps']) != 20:
            raise ValueError(f'{directory}: expected a 20 fps LeRobot v2 dataset')
        tasks = {int(row['task_index']): row['task'] for row in
                 (json.loads(line) for line in (directory / 'meta/tasks.jsonl').read_text().splitlines() if line)}
        meta = sorted((json.loads(line) for line in (directory / 'meta/episodes.jsonl').read_text().splitlines()
                       if line), key=lambda row: int(row['episode_index']))
        chunk = int(info['chunks_size'])

        def read(row):
            e, length = int(row['episode_index']), int(row['length'])
            path = directory / info['data_path'].format(episode_chunk=e // chunk, episode_index=e)
            table = pq.read_table(path, columns=['action', 'observation.state', 'annotation.human.coarse_action',
                                                 'task_index', 'episode_index', 'timestamp'])
            if table.num_rows != length or not (table.column('episode_index').to_numpy() == e).all():
                raise ValueError(f'{path}: unexpected rows')
            text = table.column('annotation.human.coarse_action').to_numpy().astype(np.int64)
            if not (table.column('task_index').to_numpy() == text).all():
                raise ValueError(f'{path}: task_index differs from the coarse-action annotation')
            if not np.allclose(table.column('timestamp').to_numpy(), np.arange(length) / 20, atol=2e-3, rtol=0):
                raise ValueError(f'{path}: frames are not contiguous')
            return vectors(table, 'action', np.float64), vectors(table, 'observation.state', np.float64), text

        with ThreadPoolExecutor(threads) as pool:
            for row, (act, st, text) in zip(meta, pool.map(read, meta)):
                parts.append((act[:, GR1_SELECT], st[:, GR1_SELECT], [prompts(tasks[int(t)]) for t in text]))
                lengths.append(int(row['length']))
                e = int(row['episode_index'])
                episodes.append(dict(folder=folder, episode_index=e, video=str(directory / info['video_path'].format(
                    episode_chunk=e // chunk, video_key=GR1_CAMERA, episode_index=e))))
    actions = np.concatenate([p[0] for p in parts])
    states = np.concatenate([p[1] for p in parts])
    stats = {}
    for kind, x in (('action', actions), ('state', states)):
        q = np.quantile(x, [0.01, 0.99], axis=0)
        stats[kind] = {'default': dict(global_min=x.min(0).tolist(), global_max=x.max(0).tolist(),
                                       global_mean=x.mean(0).tolist(), global_std=x.std(0).tolist(),
                                       global_q01=q[0].tolist(), global_q99=q[1].tolist())}
    return (actions.astype(np.float32), states.astype(np.float32), np.concatenate([p[2] for p in parts]),
            prompts.prompts, lengths, episodes, stats, None)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--benchmark', required=True, choices=('libero', 'robotwin', 'robocasa_gr1'))
    p.add_argument('--root', required=True, help='dataset root')
    p.add_argument('--out', required=True, help='prepared data directory')
    p.add_argument('--stats', help='RoboTwin: the published dataset_stats.json')
    a = p.parse_args()
    if a.benchmark == 'libero':
        result = prepare_libero(a.root)
    elif a.benchmark == 'robotwin':
        if not a.stats:
            p.error('--stats is required for RoboTwin')
        result = prepare_robotwin(a.root, a.stats)
    else:
        result = prepare_robocasa_gr1(a.root)
    actions, states, prompt_id, prompts, lengths, episodes, stats, extra = result
    starts = np.concatenate([[0], np.cumsum(lengths)[:-1]])
    write_prepared(a.out, a.benchmark, actions, states, prompt_id, prompts, starts, lengths, episodes, stats, extra)
    print(json.dumps(dict(benchmark=a.benchmark, windows=int(len(actions)), episodes=len(episodes),
                          prompts=len(prompts), out=a.out)))


if __name__ == '__main__':
    main()
