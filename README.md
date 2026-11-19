# LoopedWAM

Official code for **LoopedWAM: Looped World Action Models for Elastic Inference**.

[![Project Page](https://img.shields.io/badge/Project_Page-LoopedWAM-2ea44f.svg)](https://anhdao69.github.io/LoopedWAM)
[![Hugging Face Model](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-f7c843)](https://huggingface.co/anhdao69/LoopedWAM)

LoopedWAM is a compact world action model with a video expert and an action expert. Each expert stores 12 transformer
blocks and runs 30 layers by looping a shared six-block core four times (3 pre blocks, 6 core blocks x 4 loops,
3 coda blocks). The current frame is encoded once into a video key-value cache, and the action chunk is denoised
against that cache. LoopedWAM-E is a single checkpoint, distilled from LoopedWAM, that runs any of nine budgets:
K_a in {1, 2, 4} action loops and N in {2, 5, 10} denoising steps.

This repository contains the training and evaluation code of LoopedWAM and LoopedWAM-E on LIBERO, RoboTwin 2.0
and RoboCasa-GR1.

## Contents

- [Installation](#installation)
- [Checkpoints](#checkpoints)
- [Evaluation](#evaluation)
- [Training](#training)
- [Repository layout](#repository-layout)
- [Acknowledgements](#acknowledgements)
- [Citation](#citation)

## Installation

```bash
conda create -n loopedwam python=3.10 -y
conda activate loopedwam
pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
pip install -e .
```

Download Wan2.1-T2V-1.3B. LoopedWAM uses its VAE and T5 text encoder (frozen), and its DiT blocks initialize
training:

```bash
hf download Wan-AI/Wan2.1-T2V-1.3B --local-dir checkpoints/Wan2.1-T2V-1.3B
```

## Checkpoints

The released checkpoints are on Hugging Face at
[anhdao69/LoopedWAM](https://huggingface.co/anhdao69/LoopedWAM). Each folder holds `policy.pt`
and the `dataset_stats.json` used to normalize its actions and states.

```bash
hf download anhdao69/LoopedWAM --local-dir checkpoints/LoopedWAM
```

```text
checkpoints/LoopedWAM/
├── libero/{LoopedWAM, LoopedWAM-E}/
├── robotwin50/{LoopedWAM, LoopedWAM-E}/
└── robocasa_gr1/{LoopedWAM, LoopedWAM-E}/
```

## Evaluation

All evaluation scripts take `--checkpoint` (LoopedWAM or LoopedWAM-E), read `dataset_stats.json` next to it,
and encode instructions with the Wan2.1 T5 encoder (`--wan`). LoopedWAM runs 4 action loops at any number of
denoising steps (`--steps`, default 10). For LoopedWAM-E choose the budget with `--action-loops {1,2,4}` and
`--steps {2,5,10}`. Every query uses the same noise seed (`--seed`, default 42), so a query is a deterministic
function of the observation. Each script splits the work over `--gpus`, writes one result file per task, skips
finished tasks when run again, and writes `summary.json`.

### LIBERO

Install [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) and the simulator versions used for the paper:

```bash
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git third_party/LIBERO
pip install -e third_party/LIBERO
pip install -e ".[libero]"
```

```bash
python experiments/libero/eval_libero.py \
  --checkpoint checkpoints/LoopedWAM/libero/LoopedWAM/policy.pt \
  --gpus 0,1,2,3 --out evaluate_results/libero/LoopedWAM

python experiments/libero/eval_libero.py \
  --checkpoint checkpoints/LoopedWAM/libero/LoopedWAM-E/policy.pt \
  --action-loops 2 --steps 2 --gpus 0,1,2,3 --out evaluate_results/libero/LoopedWAM-E_k2_n2
```

Protocol: 4 suites x 10 tasks x the 50 official initial states, 30 settling steps, at most 400 policy steps
(Spatial, Object, Goal) or 700 (Long), 10 of the 32 predicted actions executed per query. `--seed` seeds the
environments and the policy noise.

### RoboTwin 2.0

Set up the RoboTwin harness (upstream RoboTwin at the commit used by FastWAM, with FastWAM's evaluation changes)
and link the LoopedWAM policy into it:

```bash
bash experiments/robotwin/setup_robotwin.sh
```

Then install RoboTwin's dependencies and download its assets following the
[RoboTwin documentation](https://github.com/RoboTwin-Platform/RoboTwin) (run inside `third_party/RoboTwin`).

```bash
python experiments/robotwin/eval_robotwin.py \
  --checkpoint checkpoints/LoopedWAM/robotwin50/LoopedWAM/policy.pt \
  --gpus 0,1,2,3 --out evaluate_results/robotwin/LoopedWAM

python experiments/robotwin/eval_robotwin.py \
  --checkpoint checkpoints/LoopedWAM/robotwin50/LoopedWAM-E/policy.pt \
  --action-loops 4 --steps 2 --gpus 0,1,2,3 --out evaluate_results/robotwin/LoopedWAM-E_k4_n2
```

Protocol: 50 tasks x {clean, randomized} x 100 episodes, seed 42, unseen instructions, 24 of the 32 predicted
actions executed per query.

### RoboCasa-GR1

Set up the simulator (robosuite, the RoboCasa GR1 tabletop tasks and starVLA's evaluation code at pinned
revisions, in a separate Python 3.10 environment) and the policy server's dependencies:

```bash
bash experiments/robocasa_gr1/setup_robocasa_gr1.sh
pip install -e ".[robocasa_gr1]"
```

```bash
python experiments/robocasa_gr1/eval_robocasa_gr1.py \
  --checkpoint checkpoints/LoopedWAM/robocasa_gr1/LoopedWAM/policy.pt \
  --gpus 0,1,2,3 --out evaluate_results/robocasa_gr1/LoopedWAM

python experiments/robocasa_gr1/eval_robocasa_gr1.py \
  --checkpoint checkpoints/LoopedWAM/robocasa_gr1/LoopedWAM-E/policy.pt \
  --action-loops 4 --steps 2 --gpus 0,1,2,3 --out evaluate_results/robocasa_gr1/LoopedWAM-E_k4_n2
```

Protocol (starVLA Robocasa_tabletop): 24 tasks x 50 episodes, 720 environment steps per episode, success at any
step, 12 of the 32 predicted actions executed per query, environment seed 0. The script starts one policy server
per GPU and runs the simulators in parallel (`--parallel`, default 24), rendering with EGL.

## Training

Training runs on one node with 4 H100 (80 GB) GPUs, on precomputed inputs: every window's 9-frame clip is
encoded once with the frozen VAE, and every instruction once with the T5 encoder. Each benchmark has three configs in `configs/<benchmark>/`:

| Config | LIBERO | RoboTwin 2.0 | RoboCasa-GR1 |
| --- | --- | --- | --- |
| `stage1.yaml` | 10 epochs, lr 1e-4, 5% warmup | 5 epochs, lr 1e-4, 5% warmup | 5 epochs, lr 1e-4, 5% warmup |
| `stage2.yaml` | 10 epochs, lr 3e-5, 3% warmup, new optimizer | 5 epochs, lr 3e-5, optimizer kept | 5 epochs, lr 3e-5, optimizer kept |
| `elastic.yaml` | 10 epochs | 5 epochs | 5 epochs |
| Batch size (stages / elastic) | 128 / 128 | 128 / 128 | 256 / 128 |

All runs use AdamW (betas 0.9, 0.95, weight decay 0.01), gradient clipping at 1.0, cosine decay to 1% of the peak
learning rate, FP32 weights and optimizer state with BF16 compute, and seed 42. LoopedWAM-E trains the action
expert at lr 3e-5 and the step condition at lr 1e-3 with 3% warmup. Every window's noise is a fixed function of the
seed and the window, so runs do not depend on the number of GPUs.

### 1. Data

```bash
# LIBERO (LeRobot v3.0, from FastWAM)
hf download yuanty/LIBERO-fastwam --repo-type dataset --include "lerobot_v30/**" --local-dir data/libero

# RoboTwin 2.0 (from FastWAM)
hf download yuanty/robotwin2.0-fastwam --repo-type dataset --local-dir data/robotwin2.0
cat data/robotwin2.0/robotwin2.0.tar.gz.part-* | tar -xzf - -C data/robotwin2.0

# RoboCasa-GR1 (24 tabletop tasks x 1,000 demonstrations)
python scripts/download_robocasa_gr1.py --out data/robocasa_gr1
```

### 2. Preprocessing

Index the training windows and write the normalization statistics:

```bash
python scripts/prepare_data.py --benchmark libero --root data/libero/lerobot_v30 --out data/prepared/libero
python scripts/prepare_data.py --benchmark robotwin --root data/robotwin2.0/robotwin2.0 \
  --stats data/robotwin2.0/dataset_stats.json --out data/prepared/robotwin
python scripts/prepare_data.py --benchmark robocasa_gr1 --root data/robocasa_gr1 --out data/prepared/robocasa_gr1
```

Encode the instructions and the video clips (shown for LIBERO; repeat for `robotwin` and `robocasa_gr1`):

```bash
python scripts/precompute_text_embeds.py --data data/prepared/libero --wan checkpoints/Wan2.1-T2V-1.3B
torchrun --standalone --nproc_per_node=4 scripts/precompute_latents.py --data data/prepared/libero \
  --vae checkpoints/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth
```

The latent caches take about 42 GB (LIBERO), 1.1 TB (RoboTwin 2.0) and 590 GB (RoboCasa-GR1).

### 3. LoopedWAM

```bash
torchrun --standalone --nproc_per_node=4 scripts/train.py --config configs/libero/stage1.yaml
torchrun --standalone --nproc_per_node=4 scripts/train.py --config configs/libero/stage2.yaml
```

Add `--resume` to continue an interrupted run from its `latest.pt`. Stage 1 starts from the Wan2.1 donor blocks;
stage 2 starts from the stage-1 checkpoint. The trained LoopedWAM is `runs/libero/stage2/final.pt`
(`runs/robotwin/stage2/final.pt`; `runs/robocasa_gr1/stage2/epoch_09.pt`).

### 4. LoopedWAM-E

```bash
torchrun --standalone --nproc_per_node=4 scripts/train_elastic.py --config configs/libero/elastic.yaml
```

The teacher is the stage-2 LoopedWAM of the same benchmark. Checkpoints are written to
`runs/<benchmark>/elastic/epoch_XX.pt` and can be evaluated directly.

Evaluate trained checkpoints with the scripts above, passing `--dataset-stats data/prepared/<benchmark>/dataset_stats.json`
and, to reuse the training instruction embeddings on LIBERO, `--text-cache data/prepared/libero/text`.

## Repository layout

```text
LoopedWAM/
├── configs/<benchmark>/{stage1, stage2, elastic}.yaml
├── scripts/
│   ├── download_robocasa_gr1.py
│   ├── prepare_data.py              # training windows and normalization statistics
│   ├── precompute_text_embeds.py    # T5 instruction embeddings
│   ├── precompute_latents.py        # VAE latents of every training clip
│   ├── train.py                     # LoopedWAM
│   └── train_elastic.py             # LoopedWAM-E
├── experiments/
│   ├── libero/eval_libero.py
│   ├── robotwin/                    # RoboTwin harness setup, policy and evaluation
│   └── robocasa_gr1/                # simulator setup, policy server, simulator client and evaluation
└── src/loopedwam/
    ├── models/                      # LoopedWAM, LoopedWAM-E, Wan2.1 modules
    ├── data/                        # windows, latent and text caches, data plans, normalization
    └── policy.py                    # evaluation policy shared by the three benchmarks
```

## Acknowledgements

This code builds on [FastWAM](https://github.com/yuantianyuan01/FastWAM) and
[Wan2.1](https://github.com/Wan-Video/Wan2.1). The RoboTwin evaluation uses the
[RoboTwin](https://github.com/RoboTwin-Platform/RoboTwin) harness with FastWAM's changes, and the RoboCasa-GR1
evaluation uses [starVLA](https://github.com/starVLA/starVLA)'s Robocasa_tabletop evaluation with
[RoboCasa GR1 tabletop tasks](https://github.com/robocasa/robocasa-gr1-tabletop-tasks). We thank the authors for
releasing their code, data and assets.

## Citation

```bibtex
@article{dao2026loopedwam,
  title   = {LoopedWAM: Looped World Action Models for Elastic Inference},
  author  = {Dao, Anh and Liu, Xinxin and Yu, Hang and
             Chan Santiago, Jeffrey A. and Shang, Yuzhang},
  year    = {2026}
}
```
