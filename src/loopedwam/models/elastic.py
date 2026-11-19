"""LoopedWAM-E: one action expert that runs at any action-loop depth K_a in {1, 2, 4} and any number of denoising
steps N in {2, 5, 10}, distilled from a frozen LoopedWAM teacher.

The video expert, its KV cache and the proprio encoder are the teacher's (frozen). The student is a trainable copy of
the teacher's action expert plus a step condition: a small MLP of the step size d = 1/N whose output is added to the
action expert's time modulation. With K_a < 4 action loops the student runs the pre blocks, the last K_a core loops
and the coda blocks, each against the matching virtual layer of the teacher's video cache.
"""
from __future__ import annotations

import copy
import hashlib
import random
from functools import partial

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .loop_mot import LoopMoT
from .loopedwam import ACTION_HORIZON, ELASTIC_FORMATS, FORMATS, LoopedWAM, load_checkpoint, masked_action_loss
from .wan_video_dit import flash_attention, modulate, sinusoidal_embedding_1d

# (K_a, ..., K_a) for N steps: the nine trained depth/step settings, named flat_k{K_a}_n{N}.
SCHEDULES = {f'flat_k{k}_n{n}': (k,) * n for k in (1, 2, 4) for n in (2, 5, 10)}
FLAT9 = tuple(SCHEDULES)
LOSS_WEIGHTS = dict(local=1.0, end=0.5, grip=0.1, anchor=0.5)


def balanced_schedule_sequence(seed: int, updates: int, names=FLAT9) -> list[str]:
    """One setting per optimizer update; every block of nine consecutive updates is a permutation of the nine."""
    rng = random.Random(f"elastic-balanced-v1:{seed}:{','.join(names)}")
    sequence = []
    while len(sequence) < updates:
        block = list(names)
        rng.shuffle(block)
        sequence.extend(block)
    return sequence[:updates]


def label_seed(label: str) -> int:
    return int(hashlib.sha256(label.encode()).hexdigest()[:15], 16)


class StepCondition(nn.Module):
    def __init__(self, width=512, freq_dim=32):
        super().__init__()
        self.width, self.freq_dim = width, freq_dim
        self.net = nn.Sequential(nn.Linear(freq_dim, 128), nn.SiLU(), nn.Linear(128, 6 * width))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, d):
        return self.net(sinusoidal_embedding_1d(self.freq_dim, d)).unflatten(-1, (6, self.width))


def cache_text_kv(expert, context):
    return [(b.cross_attn.norm_k(b.cross_attn.k(context)), b.cross_attn.v(context)) for b in expert.blocks]


def _block(mot, index, slot, x, freqs, tmod, context_mask, keys, values, mask, text_kv):
    """One action block: self-attention over [video cache slot, actions], cross-attention with cached text K/V."""
    block = mot.mixtures['action'].blocks[index]
    io = mot._build_expert_attention_io(mot.mixtures['action'], block, x, freqs, tmod)
    mixed = mot._mixed_attention(io[0], torch.cat((keys[slot], io[1]), 1), torch.cat((values[slot], io[2]), 1), mask)
    x = block.gate(io[3], io[4], block.self_attn.o(mixed))
    ca = block.cross_attn
    q = ca.norm_q(ca.q(block.norm3(x)))
    k, v = text_kv[index]
    x = x + ca.o(flash_attention(q, k, v, ca.num_heads, ctx_mask=context_mask.unsqueeze(1)))
    return block.gate(x, io[7], block.ffn(modulate(block.norm2(x), io[5], io[6])))


def action_core(mot, tokens, freqs, tmod, context_mask, keys, values, mask, *, loops, text_kv,
                checkpoint_blocks=False):
    """pre(3) -> the last `loops` core loops -> coda(3), reading the video cache slots of the same virtual layers."""
    slots = list(range(3)) + [3 + 6 * (4 - loops + r) + j for r in range(loops) for j in range(6)] + list(range(27, 30))
    schedule = mot.virtual_schedule()
    x = tokens
    for slot in slots:
        fn = partial(_block, mot, schedule[slot][1], slot)
        args = (x, freqs, tmod, context_mask, keys, values, mask, text_kv)
        x = checkpoint(fn, *args, use_reentrant=False) if checkpoint_blocks and torch.is_grad_enabled() else fn(*args)
    return x


@torch.no_grad()
def teacher_target(velocity, z, tau, next_tau):
    """Average teacher velocity over the student macrostep [tau, next_tau], on the teacher's 10-step grid."""
    steps = round((tau - next_tau) * 10)
    x = z.detach()
    grid = torch.linspace(1., 0., 11, device=z.device, dtype=z.dtype)
    start = round((1 - tau) * 10)
    if steps == 1:
        return velocity(x, grid[start].expand(len(x))).detach()
    for i in range(steps):
        x = x + (grid[start + i + 1] - grid[start + i]) * velocity(x, grid[start + i].expand(len(x)))
    return ((x - z.detach()) / (grid[start + steps] - grid[start])).detach()


def gripper_command(actions, scale, offset):
    """LIBERO gripper command from normalized actions; positive closes, negative opens."""
    return 1 - 2 * ((actions[..., 6] - offset[6]) / scale[6])


def endpoint_losses(student, teacher, scale, offset, prefix, gripper):
    """MSE to the teacher's 10-step endpoint over the executed prefix of the chunk, plus (LIBERO only) a hinge on
    the binarized gripper command."""
    end = (student[:, :prefix].float() - teacher[:, :prefix].float()).square().mean((1, 2))
    if not gripper:
        return end, end.new_zeros(end.shape)
    target = gripper_command(teacher[:, :prefix], scale, offset)
    sign = torch.where(target >= 0, 1., -1.)
    grip = F.softplus(-5 * sign * gripper_command(student[:, :prefix], scale, offset)).mean(1)
    return end, grip


class LoopedWAME(nn.Module):
    """Frozen LoopedWAM teacher + trainable elastic student action expert."""

    def __init__(self, parent: LoopedWAM, *, end_prefix=10, gripper_loss=True, checkpoint_blocks=False, seed=42):
        super().__init__()
        self.parent = parent.requires_grad_(False).eval()
        self.end_prefix, self.gripper_loss, self.checkpoint_blocks = end_prefix, gripper_loss, checkpoint_blocks
        action = copy.deepcopy(parent.action_expert).requires_grad_(True)
        width = action.hidden_dim
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(label_seed(f'elastic-common-v1:step_condition:{seed}'))
            self.step_condition = StepCondition(width)
        self.step_condition.to(parent.device)
        self.student = LoopMoT({'video': parent.video_expert, 'action': action})

    @property
    def student_action(self):
        return self.student.mixtures['action']

    def train(self, mode=True):
        super().train(mode)
        self.parent.eval()
        self.student.mixtures['video'].eval()
        return self

    def optimizer_groups(self):
        step = list(self.step_condition.parameters())
        expert = list(self.student_action.parameters())
        return [dict(params=step, lr=1e-3, base_lr=1e-3, weight_decay=0.),
                dict(params=expert, lr=3e-5, base_lr=3e-5, weight_decay=.01)]

    @torch.no_grad()
    def prefill(self, sample):
        parent = self.parent
        context, mask = parent._append_proprio_to_context(sample['context'], sample['context_mask'],
                                                          sample['proprio'][:, 0])
        latents = sample['first_frame_latents']
        video = parent.video_expert.prepare(x=latents, timestep=latents.new_zeros(len(latents)), context=context,
                                            context_mask=mask, action=None, fuse_vae_embedding_in_latents=True)
        n = video[0].shape[1]
        attention = parent._build_mot_attention_mask(n, ACTION_HORIZON, video[9], latents.device)
        keys, values = parent.mot.prefill_video_cache_tensor(video[0], video[5], video[2], video[3], video[4],
                                                             attention[:n, :n])
        return context, mask, keys, values, attention[n:, :]

    @staticmethod
    def conditioning(expert, context, mask):
        embedded = expert.text_embedding(context)
        return embedded, mask[:, None, :].expand(-1, ACTION_HORIZON, -1), cache_text_kv(expert, embedded)

    def velocity(self, z, tau, conditioning, cache, *, teacher=False, loops=4, d=0., native_timestep=None):
        mot = self.parent.mot if teacher else self.student
        expert = mot.mixtures['action']
        _, mask, text_kv = conditioning
        native_timestep = tau * 1000 if native_timestep is None else native_timestep
        t = expert.time_embedding(sinusoidal_embedding_1d(expert.freq_dim, native_timestep))
        tmod = expert.time_projection(t).unflatten(1, (6, expert.hidden_dim))
        if not teacher:
            tmod = tmod + self.step_condition(tau.new_full(tau.shape, d))
        out = action_core(mot, expert.action_encoder(z), expert.get_freqs(z.shape[1]), tmod, mask, *cache,
                          loops=loops, text_kv=text_kv, checkpoint_blocks=self.checkpoint_blocks and not teacher)
        return expert.post(out)

    def forward(self, sample, schedule, noise, anchor_noise, anchor_tau):
        """Summed (over valid samples) distillation loss for one depth/step setting `schedule` = (K_a,) * N."""
        context, mask, keys, values, attention = self.prefill(sample)
        cache = (keys, values, attention)
        with torch.no_grad():
            teacher_condition = self.conditioning(self.parent.action_expert, context, mask)

        def teacher(z, t):
            return self.velocity(z, t, teacher_condition, cache, teacher=True)

        student_condition = self.conditioning(self.student_action, context, mask)
        # The teacher's 10-step trajectory from the shared initial noise.
        states, velocities, endpoint = [], [], noise.detach()
        grid = torch.linspace(1., 0., 11, device=noise.device, dtype=noise.dtype)
        with torch.no_grad():
            for j in range(10):
                states.append(endpoint)
                velocities.append(teacher(endpoint, grid[j].expand(len(endpoint))))
                endpoint = endpoint + (grid[j + 1] - grid[j]) * velocities[-1]
        n = len(schedule)
        z = noise
        local = z.new_zeros(len(z))
        for j, k in enumerate(schedule):
            if j <= n - 3:
                z = z.detach()   # gradients flow through the last two student steps only
            tau, next_tau = 1 - j / n, 1 - (j + 1) / n
            start, stop = round(j * 10 / n), round((j + 1) * 10 / n)
            current_tau = grid[start].expand(len(z))
            if j == 0:
                # The first macrostep starts from the shared noise: reuse the teacher trajectory computed above.
                target = (velocities[0] if stop == 1 else
                          ((states[stop] - z.detach()) / (grid[stop] - grid[start])).detach())
            else:
                target = teacher_target(teacher, z, tau, next_tau)
            v = self.velocity(z, current_tau, student_condition, cache, loops=k, d=1 / n)
            local = local + masked_action_loss(v, target, sample['action_is_pad']) / n
            z = z + (grid[stop] - grid[start]) * v
        end, grip = endpoint_losses(z, endpoint, sample['action_scale'], sample['action_offset'],
                                    prefix=self.end_prefix, gripper=self.gripper_loss)
        # Anchor: the native flow-matching loss at full depth (d = 0).
        clean = sample['action']
        noisy = anchor_tau[:, None, None] * anchor_noise + (1 - anchor_tau[:, None, None]) * clean
        v = self.velocity(noisy, anchor_tau, student_condition, cache, loops=4, d=0.)
        anchor = masked_action_loss(v, anchor_noise - clean, sample['action_is_pad'])
        anchor = anchor * self.parent.train_action_scheduler.training_weight(anchor_tau * 1000)
        components = dict(local=local, end=end, grip=grip, anchor=anchor)
        loss = sum(LOSS_WEIGHTS[key] * value for key, value in components.items())
        valid = sample['sample_valid'].to(loss)
        logs = {key: (value.detach() * valid).sum() for key, value in components.items()}
        return (loss * valid).sum(), logs

    @torch.no_grad()
    def rollout(self, sample, schedule, noise):
        """Normalized action chunks [B, 32, A] with N = len(schedule) Euler steps at depths `schedule`."""
        context, mask, keys, values, attention = self.prefill(sample)
        condition = self.conditioning(self.student_action, context, mask)
        z, n = noise, len(schedule)
        times, deltas = self.parent.infer_action_scheduler.build_inference_schedule(n, z.device, z.dtype,
                                                                                    shift_override=1.)
        for j, k in enumerate(schedule):
            tau = (times[j] / 1000).expand(len(z))
            v = self.velocity(z, tau, condition, (keys, values, attention), loops=k, d=1 / n,
                              native_timestep=times[j].expand(len(z)))
            z = z + v * deltas[j]
        return z

    @torch.no_grad()
    def infer_action(self, image, proprio, context, context_mask, seed=42, action_loops=4, steps=10):
        """Normalized 32-action chunk [32, A] for one observation, at depth `action_loops` with `steps` steps.

        Call under BF16 autocast. image: [3, H, W] in [-1, 1]; proprio: [D] normalized state;
        context/context_mask: T5 text [128, 4096] / [128].
        """
        self.eval()
        parent = self.parent
        generator = torch.Generator(device='cpu').manual_seed(int(seed))
        noise = torch.randn((1, ACTION_HORIZON, parent.action_dim), generator=generator, dtype=torch.float32)
        image = image.unsqueeze(0) if image.ndim == 3 else image
        latents = parent.encode_image(image.to(device=parent.device, dtype=torch.float32))
        sample = dict(first_frame_latents=latents,
                      proprio=proprio.reshape(1, 1, -1).to(device=parent.device, dtype=torch.float32),
                      context=(context.unsqueeze(0) if context.ndim == 2 else context).to(parent.device, torch.float32),
                      context_mask=(context_mask.unsqueeze(0) if context_mask.ndim == 1 else context_mask).to(
                          parent.device, torch.bool))
        schedule = SCHEDULES[f'flat_k{int(action_loops)}_n{int(steps)}']
        return self.rollout(sample, schedule, noise.to(parent.device))[0].float().cpu()


# ---------------------------------------------------------------------- checkpoints
def load_elastic(path, vae_path, *, device='cuda'):
    """Load a LoopedWAM-E checkpoint -> (LoopedWAME, payload)."""
    payload = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
    if 'student_action' not in payload or 'step_condition' not in payload:
        raise ValueError(f'{path}: not a LoopedWAM-E checkpoint')
    parent, _ = load_checkpoint(path, vae_path, device=device)
    config = payload.get('transport_config', {})
    policy = LoopedWAME(parent, end_prefix=config.get('end_prefix', 10),
                        gripper_loss=config.get('gripper_loss', True)).to(device)
    policy.student_action.load_state_dict(payload['student_action'], strict=True)
    policy.step_condition.load_state_dict(payload['step_condition'], strict=True)
    return policy.eval().requires_grad_(False), payload


def elastic_payload(policy, teacher_payload, benchmark, step, config, training_state):
    """LoopedWAM-E checkpoint: the teacher's payload with the trained student written into its action expert."""
    student = {k: v.detach().cpu().clone() for k, v in policy.student_action.state_dict().items()}
    mot = dict(teacher_payload['mot'])
    for k, v in student.items():
        mot['mixtures.action.' + k] = v
    payload = {k: v for k, v in teacher_payload.items() if k not in ('mot', 'optimizer', 'training_state', 'step')}
    payload.update(format_version=ELASTIC_FORMATS[benchmark], mot=mot, step=step, transport_config=config,
                   student_action=student, inference_action_format='unmerged-exact', transport={},
                   step_condition={k: v.detach().cpu().clone() for k, v in policy.step_condition.state_dict().items()},
                   training_state=training_state)
    if ELASTIC_FORMATS[benchmark] != FORMATS[benchmark]:
        payload['base_format_version'] = FORMATS[benchmark]
    return payload
