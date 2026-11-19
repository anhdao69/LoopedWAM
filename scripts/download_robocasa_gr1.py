#!/usr/bin/env python3
"""Download the RoboCasa-GR1 training data: the 24 `gr1_unified.*_1000` folders (1,000 demonstrations each) of
nvidia/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim at a pinned revision.

    python scripts/download_robocasa_gr1.py --out data/robocasa_gr1

The repository is too large to list at once, so each folder is listed and its files are downloaded one by one.
Files already present with the expected size are skipped, so the script can be run again to resume.
"""
from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

from loopedwam.data.robocasa_gr1 import DATASET_REPO, DATASET_REVISION, GR1_FOLDERS


def retry(fn, attempts=20):
    for attempt in range(attempts):
        try:
            return fn()
        except Exception:
            if attempt == attempts - 1:
                raise
            time.sleep(min(60, 5 * (attempt + 1)))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--out', required=True)
    p.add_argument('--workers', type=int, default=16)
    a = p.parse_args()
    out, api = Path(a.out), HfApi()
    files = []
    for folder in GR1_FOLDERS:
        entries = retry(lambda: list(api.list_repo_tree(DATASET_REPO, repo_type='dataset', revision=DATASET_REVISION,
                                                         path_in_repo=folder, recursive=True)))
        files += [(e.path, e.size) for e in entries if getattr(e, 'size', None) is not None]
    todo = [(path, size) for path, size in files if not ((out / path).is_file() and (out / path).stat().st_size == size)]
    print(json.dumps(dict(files=len(files), gigabytes=round(sum(s for _, s in files) / 1e9, 1), todo=len(todo))),
          flush=True)

    def fetch(path, size):
        retry(lambda: hf_hub_download(DATASET_REPO, path, repo_type='dataset', revision=DATASET_REVISION,
                                      local_dir=str(out)))
        if (out / path).stat().st_size != size:
            raise RuntimeError(f'{path}: size mismatch')

    with ThreadPoolExecutor(a.workers) as pool:
        for done, future in enumerate(as_completed([pool.submit(fetch, *f) for f in todo]), 1):
            future.result()
            if done % 1000 == 0 or done == len(todo):
                print(json.dumps(dict(downloaded=done, of=len(todo))), flush=True)
    for folder in GR1_FOLDERS:
        meta = out / folder / 'meta' / 'episodes.jsonl'
        episodes = [line for line in meta.read_text().splitlines() if line.strip()]
        videos = list((out / folder / 'videos').glob('chunk-*/observation.images.ego_view/episode_*.mp4'))
        if len(episodes) != 1000 or len(videos) != 1000:
            raise SystemExit(f'{folder}: expected 1000 episodes and videos')
    print(json.dumps(dict(event='done', folders=len(GR1_FOLDERS))), flush=True)


if __name__ == '__main__':
    main()
