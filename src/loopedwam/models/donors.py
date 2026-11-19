"""Initialization of LoopedWAM from the native Wan2.1-T2V-1.3B weights (CPU, FP32, deterministic).

Video expert: the 30 native blocks with every FFN pruned from 8960 to 6144 hidden units (evenly spaced, paired
rows/columns); blocks 0-2, 9-14 and 27-29 become the 12 physical blocks (pre 3, core 6, coda 3).
Action expert: every native backbone tensor linearly resized to the 512-wide expert (sequential 1D linear
interpolation with aligned corners, then sqrt(fan-in ratio) scaling), with the same 12 donor blocks.
Action encoder, action head and proprio encoder are new default nn.Linear layers drawn from a fixed seed.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from .action_dit import ActionDiT
from .wan_video_dit import WanVideoDiT

DONOR_LAYERS = (0, 1, 2, 9, 10, 11, 12, 13, 14, 27, 28, 29)
TEXT_DIM = 4096
NATIVE_CONFIG = dict(model_type='t2v', dim=1536, ffn_dim=8960, freq_dim=256, in_dim=16, out_dim=16,
                     num_heads=12, num_layers=30, eps=1e-6)


def expert_configs(action_dim: int, num_layers: int = 12) -> tuple[dict, dict]:
    common = dict(text_dim=TEXT_DIM, freq_dim=256, eps=1e-6, num_heads=12, attn_head_dim=128, num_layers=num_layers)
    video = dict(**common, hidden_dim=1536, ffn_dim=6144, in_dim=16, out_dim=16, patch_size=(1, 2, 2),
                 has_image_input=False, seperated_timestep=True, video_attention_mask_mode='first_frame_causal')
    action = dict(**common, hidden_dim=512, ffn_dim=2048, action_dim=action_dim)
    return video, action


def ffn_indices(native_width: int = 8960, target_width: int = 6144) -> torch.Tensor:
    return torch.linspace(0, native_width - 1, target_width, dtype=torch.float64).round().long()


def resize_action_tensor(source: torch.Tensor, target_shape) -> torch.Tensor:
    """Sequential per-axis linear interpolation (align_corners=True), then last-axis sqrt(fan-in) scaling."""
    source = source.to(device='cpu', dtype=torch.float32)
    if tuple(source.shape) == tuple(target_shape):
        return source
    result = source
    while result.ndim < len(target_shape):
        result = result.unsqueeze(0)
    while result.ndim > len(target_shape):
        result = result.squeeze(0)
    for dim, size in enumerate(target_shape):
        if result.shape[dim] == size:
            continue
        moved = result.movedim(dim, -1).contiguous()
        flat = moved.reshape(-1, 1, moved.shape[-1])
        resized = F.interpolate(flat, size=size, mode='linear', align_corners=True)
        result = resized.reshape(*moved.shape[:-1], size).movedim(-1, dim).contiguous()
    if source.ndim >= 2 and source.shape[-1] != target_shape[-1]:
        result = result * (source.shape[-1] / target_shape[-1]) ** 0.5
    return result


def read_native_wan(wan_dir) -> dict:
    """All tensors of the native Wan2.1-T2V-1.3B DiT (`diffusion_pytorch_model*.safetensors`)."""
    from safetensors import safe_open
    wan_dir = Path(wan_dir)
    config = json.loads((wan_dir / 'config.json').read_text())
    for key, expected in NATIVE_CONFIG.items():
        if config.get(key) != expected:
            raise ValueError(f'{wan_dir}: expected Wan2.1-T2V-1.3B ({key}={expected!r}), got {config.get(key)!r}')
    paths = sorted(wan_dir.glob('diffusion_pytorch_model*.safetensors'))
    if not paths:
        raise FileNotFoundError(f'No diffusion_pytorch_model*.safetensors in {wan_dir}')
    state = {}
    for path in paths:
        with safe_open(str(path), framework='pt', device='cpu') as handle:
            for raw in handle.keys():
                key = raw[6:] if raw.startswith('model.') else raw
                if key in state:
                    raise ValueError(f'Duplicate tensor {key}')
                state[key] = handle.get_tensor(raw)
    return state


def _select_layers(state: dict) -> dict:
    lookup = {source: target for target, source in enumerate(DONOR_LAYERS)}
    result = {}
    for key, value in state.items():
        if not key.startswith('blocks.'):
            result[key] = value
            continue
        _, index, suffix = key.split('.', 2)
        if int(index) in lookup:
            result[f'blocks.{lookup[int(index)]}.{suffix}'] = value
    return result


def wan_donor_states(wan_dir) -> tuple[dict, dict]:
    """Video and action expert backbone tensors (12 blocks each) from the native Wan2.1 weights."""
    native = read_native_wan(wan_dir)
    video_cfg, action_cfg = expert_configs(action_dim=7, num_layers=30)
    with torch.device('meta'):
        native_model = WanVideoDiT(**{**video_cfg, 'ffn_dim': 8960})
        action_model = ActionDiT(**action_cfg)
    native_shapes = {k: tuple(v.shape) for k, v in native_model.state_dict().items()}
    action_shapes = {k: tuple(v.shape) for k, v in action_model.state_dict().items()}
    if set(native) != set(native_shapes) or any(tuple(native[k].shape) != s for k, s in native_shapes.items()):
        raise ValueError('The native checkpoint does not match the Wan2.1-T2V-1.3B DiT')
    indices = ffn_indices()
    video, action = {}, {}
    for key in sorted(native_shapes):
        tensor = native[key].detach().to(device='cpu', dtype=torch.float32).contiguous()
        if key in ActionDiT.backbone_key_set(action_shapes):
            # The action expert is derived from the native tensor, before any FFN pruning.
            action[key] = resize_action_tensor(tensor, action_shapes[key])
        if key.startswith('blocks.') and key.endswith(('ffn.0.weight', 'ffn.0.bias')):
            video[key] = tensor.index_select(0, indices)
        elif key.startswith('blocks.') and key.endswith('ffn.2.weight'):
            video[key] = tensor.index_select(1, indices)
        else:
            video[key] = tensor
    return _select_layers(video), _select_layers(action)


def fresh_interfaces(action_dim: int, proprio_dim: int, hidden_dim: int = 512, seed: int = 42):
    """Action encoder, action head and proprio encoder: default nn.Linear layers drawn from `seed`."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        encoder = nn.Linear(action_dim, hidden_dim)
        head = nn.Linear(hidden_dim, action_dim)
        proprio = nn.Linear(proprio_dim, TEXT_DIM)
    return encoder, head, proprio


def load_wan21_vae(path, *, device='cpu', dtype=torch.float32):
    """The native 16-channel Wan2.1 VAE (frozen)."""
    from .wan_video_vae import WanVideoVAE
    raw = torch.load(str(path), map_location='cpu', weights_only=True)
    if 'model_state' in raw:
        raw = raw['model_state']
    state = {key if key.startswith('model.') else f'model.{key}': value for key, value in raw.items()}
    vae = WanVideoVAE(z_dim=16)
    vae.load_state_dict(state, strict=True)
    return vae.eval().requires_grad_(False).to(device=device, dtype=dtype)
