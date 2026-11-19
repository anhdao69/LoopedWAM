"""LoopedWAM: a FastWAM-style world action model with a looped (shared-depth) video/action transformer.

Training: flow matching on the future video latents (shift 5) and the 32-action chunk (shift 1), conditioned on the
clean current-frame latents, the T5 instruction and a proprioception token. Inference: the current frame is encoded
once into a 30-layer video KV cache, then the action chunk is denoised with Euler steps (CFG 1).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .donors import TEXT_DIM, expert_configs, fresh_interfaces, load_wan21_vae, wan_donor_states
from .loop_mot import LoopMoT
from .action_dit import ActionDiT
from .scheduler import WanContinuousFlowMatchScheduler
from .wan_video_dit import WanVideoDiT, precompute_freqs_cis, precompute_freqs_cis_3d

ACTION_HORIZON = 32
# Checkpoint format names, per benchmark (LoopedWAM / LoopedWAM-E).
FORMATS = {'libero': 'loopwam-s-v1', 'robotwin': 'loopwam-s-robot-v1', 'robocasa_gr1': 'loopwam-s-gr1-v1'}
ELASTIC_FORMATS = {'libero': 'loopwam-s-v1', 'robotwin': 'loopwam-s-robot-rt-v1', 'robocasa_gr1': 'loopwam-s-gr1-rt-v1'}


def masked_action_loss(pred, target, is_pad=None):
    """Per-sample mean squared error over the valid (unpadded) actions."""
    error = (pred.float() - target.float()).square().mean(-1)
    if is_pad is None:
        return error.mean(-1)
    valid = (~is_pad.bool()).to(error)
    return (error * valid).sum(-1) / valid.sum(-1).clamp_min(1)


class LoopedWAM(nn.Module):
    def __init__(self, video_expert: WanVideoDiT, action_expert: ActionDiT, vae, proprio_dim: int,
                 checkpoint_blocks: bool = False):
        super().__init__()
        self.video_expert = video_expert
        self.action_expert = action_expert
        self.mot = LoopMoT({'video': video_expert, 'action': action_expert}, checkpoint_blocks=checkpoint_blocks)
        self.vae = vae.requires_grad_(False).eval()
        self.proprio_dim = int(proprio_dim)
        self.proprio_encoder = nn.Linear(self.proprio_dim, TEXT_DIM)
        self.train_video_scheduler = WanContinuousFlowMatchScheduler(num_train_timesteps=1000, shift=5.0)
        self.train_action_scheduler = WanContinuousFlowMatchScheduler(num_train_timesteps=1000, shift=1.0)
        self.infer_action_scheduler = WanContinuousFlowMatchScheduler(num_train_timesteps=1000, shift=1.0)

    @property
    def device(self):
        return self.proprio_encoder.weight.device

    @property
    def action_dim(self):
        return self.action_expert.action_dim

    def train(self, mode=True):
        super().train(mode)
        self.vae.eval()
        return self

    def policy_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    def forward(self, batch, noise):
        return self.training_loss(batch, noise)

    # ------------------------------------------------------------------ inputs
    def _append_proprio_to_context(self, context, context_mask, proprio):
        token = self.proprio_encoder(proprio.to(device=self.device, dtype=context.dtype).unsqueeze(1)).to(dtype=context.dtype)
        token_mask = torch.ones((context_mask.shape[0], 1), dtype=torch.bool, device=context_mask.device)
        return torch.cat([context, token], dim=1), torch.cat([context_mask, token_mask], dim=1)

    @torch.no_grad()
    def _build_mot_attention_mask(self, video_seq_len, action_seq_len, video_tokens_per_frame, device):
        """Video tokens attend causally by frame (first frame to itself), actions to the first frame and actions."""
        total = video_seq_len + action_seq_len
        mask = torch.zeros((total, total), dtype=torch.bool, device=device)
        mask[:video_seq_len, :video_seq_len] = self.video_expert.build_video_to_video_mask(
            video_seq_len=video_seq_len, video_tokens_per_frame=video_tokens_per_frame, device=device)
        mask[video_seq_len:, video_seq_len:] = True
        mask[video_seq_len:, :min(video_tokens_per_frame, video_seq_len)] = True
        return mask

    @torch.no_grad()
    def encode_video(self, video):
        """[B, 3, T, H, W] in [-1, 1] -> FP32 latents, one independent VAE pass per clip."""
        dtype = next(self.vae.parameters()).dtype
        latents = self.vae.model.encode(video.to(device=self.device, dtype=dtype), self.vae.scale)
        self.vae.model.clear_cache()
        return latents.to(torch.float32)

    @torch.no_grad()
    def encode_image(self, image):
        """[B, 3, H, W] in [-1, 1] -> FP32 latents [B, 16, 1, H/8, W/8]."""
        if image.ndim == 3:
            image = image.unsqueeze(0)
        return self.encode_video(image.unsqueeze(2))

    # ------------------------------------------------------------------ training
    def _video_loss_per_sample(self, pred, target, image_is_pad):
        """MSE per future latent frame, averaged over the frames that are not padding."""
        loss = F.mse_loss(pred.float(), target.float(), reduction='none').mean(dim=(1, 3, 4))
        factor = int(self.vae.temporal_downsample_factor)
        tail_is_pad = image_is_pad[:, 1:].view(image_is_pad.shape[0], -1, factor).all(dim=2)
        valid = (~tail_is_pad).to(device=loss.device, dtype=loss.dtype)
        return (loss * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1.0)

    def training_loss(self, batch, noise):
        """Summed (over the batch) flow-matching loss of the video and action streams.

        batch: latents [B,16,T,h,w] (clean clip latents), action [B,32,A], proprio [B,D] (current state),
               context [B,128,4096], context_mask [B,128], action_is_pad [B,32], image_is_pad [B,9]
        noise: video [B,16,T,h,w], action [B,32,A], t_video [B], t_action [B]
        """
        clean_video = batch['latents']
        clean_action = batch['action'].to(device=self.device, dtype=torch.float32)
        context = batch['context'].to(device=self.device, dtype=torch.float32)
        context_mask = batch['context_mask'].to(device=self.device, dtype=torch.bool)
        context, context_mask = self._append_proprio_to_context(
            context, context_mask, batch['proprio'].to(device=self.device, dtype=torch.float32))
        t_video, t_action = noise['t_video'], noise['t_action']
        noisy_video = self.train_video_scheduler.add_noise(clean_video, noise['video'], t_video)
        noisy_video = torch.cat([clean_video[:, :, :1], noisy_video[:, :, 1:]], dim=2)
        noisy_action = self.train_action_scheduler.add_noise(clean_action, noise['action'], t_action)
        target_video, target_action = noise['video'] - clean_video, noise['action'] - clean_action
        _, ph, pw = self.video_expert.patch_size
        frames, height, width = clean_video.shape[-3:]
        tokens_per_frame = (height // ph) * (width // pw)
        attention_mask = self._build_mot_attention_mask(frames * tokens_per_frame, noisy_action.shape[1],
                                                        tokens_per_frame, clean_video.device)
        self.mot.observation_tokens = tokens_per_frame

        video = self.video_expert.prepare(x=noisy_video, timestep=t_video, context=context, context_mask=context_mask,
                                          action=None, fuse_vae_embedding_in_latents=True)
        action = self.action_expert.prepare(action_tokens=noisy_action, timestep=t_action, context=context,
                                            context_mask=context_mask)
        video_out, action_out = self.mot.forward_joint(
            video_tokens=video[0], action_tokens=action[0], video_freqs=video[5], action_freqs=action[5],
            video_t_mod=video[2], action_t_mod=action[2], video_context=video[3], video_context_mask=video[4],
            action_context=action[3], action_context_mask=action[4], attention_mask=attention_mask)
        pred_video = self.video_expert.post(video_out, video[1], video[6], video[7], video[8])
        pred_action = self.action_expert.post(action_out)

        raw_video = self._video_loss_per_sample(pred_video[:, :, 1:], target_video[:, :, 1:], batch['image_is_pad'])
        raw_action = masked_action_loss(pred_action, target_action, batch['action_is_pad'])
        loss_video = (raw_video * self.train_video_scheduler.training_weight(t_video)).mean()
        loss_action = (raw_action * self.train_action_scheduler.training_weight(t_action)).mean()
        b = clean_action.shape[0]
        logs = dict(video=loss_video.detach() * b, action=loss_action.detach() * b)
        return (loss_video + loss_action) * b, logs

    # ------------------------------------------------------------------ inference
    def prefill(self, latents, context, context_mask):
        """Video KV cache of the clean current frame, plus the action rows of the attention mask."""
        timestep = torch.zeros((latents.shape[0],), dtype=latents.dtype, device=self.device)
        tokens, _, t_mod, video_context, video_mask, freqs, _, _, _, per_frame = self.video_expert.prepare(
            x=latents, timestep=timestep, context=context, context_mask=context_mask, action=None,
            fuse_vae_embedding_in_latents=True)
        n = int(tokens.shape[1])
        attention = self._build_mot_attention_mask(n, ACTION_HORIZON, per_frame, tokens.device)
        keys, values = self.mot.prefill_video_cache_tensor(video_tokens=tokens, video_freqs=freqs, video_t_mod=t_mod,
                                                           video_context=video_context, video_context_mask=video_mask,
                                                           video_attention_mask=attention[:n, :n])
        return keys, values, attention[n:, :]

    def _denoise_action_with_video_cache(self, latents_action, timestep_action, context, context_mask,
                                         video_cache_k, video_cache_v, action_attention_mask):
        tokens, _, t_mod, action_context, action_mask, freqs = self.action_expert.prepare(
            action_tokens=latents_action, timestep=timestep_action, context=context, context_mask=context_mask)
        tokens = self.mot.forward_action_with_video_cache_tensor(
            action_tokens=tokens, action_freqs=freqs, action_t_mod=t_mod, action_context=action_context,
            action_context_mask=action_mask, video_cache_k=video_cache_k, video_cache_v=video_cache_v,
            action_attention_mask=action_attention_mask)
        return self.action_expert.post(tokens)

    @torch.no_grad()
    def infer_action(self, image, proprio, context, context_mask, seed=42, num_inference_steps=10):
        """Normalized 32-action chunk [32, A] for one observation. Call under BF16 autocast.

        image: [3, H, W] in [-1, 1]; proprio: [D] normalized state; context/context_mask: T5 text [128, 4096] / [128].
        The initial noise is drawn from a CPU generator seeded with `seed`.
        """
        self.eval()
        generator = torch.Generator(device='cpu').manual_seed(int(seed))
        noise = torch.randn((1, ACTION_HORIZON, self.action_dim), generator=generator, dtype=torch.float32)
        x = noise.to(self.device)
        image = image.unsqueeze(0) if image.ndim == 3 else image
        latents = self.encode_image(image.to(device=self.device, dtype=torch.float32))
        proprio = proprio.reshape(1, -1).to(device=self.device, dtype=torch.float32)
        context = (context.unsqueeze(0) if context.ndim == 2 else context).to(device=self.device, dtype=torch.float32)
        context_mask = (context_mask.unsqueeze(0) if context_mask.ndim == 1 else context_mask).to(
            device=self.device, dtype=torch.bool)
        context, context_mask = self._append_proprio_to_context(context, context_mask, proprio)
        keys, values, action_mask = self.prefill(latents, context, context_mask)
        timesteps, deltas = self.infer_action_scheduler.build_inference_schedule(
            num_inference_steps=num_inference_steps, device=self.device, dtype=x.dtype, shift_override=None)
        for step_t, delta in zip(timesteps, deltas):
            pred = self._denoise_action_with_video_cache(
                latents_action=x, timestep_action=step_t.unsqueeze(0).to(dtype=x.dtype, device=self.device),
                context=context, context_mask=context_mask, video_cache_k=keys, video_cache_v=values,
                action_attention_mask=action_mask)
            x = self.infer_action_scheduler.step(pred, delta, x)
        return x[0].detach().to(device='cpu', dtype=torch.float32)


# ---------------------------------------------------------------------- construction and checkpoints
def build_loopedwam(action_dim, proprio_dim, vae_path, *, device='cpu', wan_dir=None, seed=42,
                    checkpoint_blocks=False):
    """A LoopedWAM with uninitialized weights (to be loaded), or initialized from Wan2.1 when `wan_dir` is given.

    Policy weights are FP32; on CUDA the frozen VAE runs in BF16.
    """
    video_cfg, action_cfg = expert_configs(action_dim)
    if wan_dir is None:
        video, action = WanVideoDiT(**video_cfg), ActionDiT(**action_cfg)
    else:
        video_state, action_state = wan_donor_states(wan_dir)
        encoder, head, proprio = fresh_interfaces(action_dim, proprio_dim, seed=seed)
        action_state = dict(action_state, **{f'action_encoder.{k}': v for k, v in encoder.state_dict().items()},
                            **{f'head.{k}': v for k, v in head.state_dict().items()})
        with torch.device('meta'):
            video, action = WanVideoDiT(**video_cfg), ActionDiT(**action_cfg)
        video.load_state_dict(video_state, strict=True, assign=True)
        action.load_state_dict(action_state, strict=True, assign=True)
        video.freqs = precompute_freqs_cis_3d(128)
        action.freqs = precompute_freqs_cis(128)
    model = LoopedWAM(video, action, load_wan21_vae(vae_path), proprio_dim, checkpoint_blocks=checkpoint_blocks)
    if wan_dir is not None:
        model.proprio_encoder.load_state_dict(proprio.state_dict(), strict=True)
    model.to(device)
    if torch.device(device).type == 'cuda':
        model.vae.to(dtype=torch.bfloat16)
    return model


def load_checkpoint(path, vae_path, *, device='cuda', checkpoint_blocks=False):
    """Load a LoopedWAM checkpoint (any benchmark) -> (model, payload)."""
    payload = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
    if payload.get('format_version') not in set(FORMATS.values()) | set(ELASTIC_FORMATS.values()):
        raise ValueError(f'{path}: not a LoopedWAM checkpoint')
    action_dim = payload['mot']['mixtures.action.head.weight'].shape[0]
    proprio_dim = payload['proprio_encoder']['weight'].shape[1]
    model = build_loopedwam(action_dim, proprio_dim, vae_path, device='cpu', checkpoint_blocks=checkpoint_blocks)
    model.mot.load_state_dict(payload['mot'], strict=True)
    model.proprio_encoder.load_state_dict(payload['proprio_encoder'], strict=True)
    model.to(device)
    if torch.device(device).type == 'cuda':
        model.vae.to(dtype=torch.bfloat16)
    return model, payload


def checkpoint_payload(model, benchmark, step, config):
    """Checkpoint dictionary in the released LoopedWAM format of `benchmark`."""
    video_cfg, action_cfg = expert_configs(model.action_dim)
    return dict(format_version=FORMATS[benchmark], version='v0',
                video_loops=4, action_loops=4, inference_loops=4, trained_max_loops=4, exit_weight_scale=1.0,
                architecture=dict(target_video_config=video_cfg, target_action_config=action_cfg),
                mot={k: v.detach().cpu() for k, v in model.mot.state_dict().items()},
                proprio_encoder={k: v.detach().cpu() for k, v in model.proprio_encoder.state_dict().items()},
                step=step, training_state=dict(config=config))
