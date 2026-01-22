# Copyright (c) OpenMMLab. All rights reserved.
# Ported from original CO-DETR repo for MMDetection v3
import logging
import math
from functools import partial
import warnings
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Union, List

from mmengine.model import BaseModule
from mmengine.runner import load_checkpoint
from mmengine.logging import print_log
from mmdet.registry import MODELS

try:
    from timm.layers import DropPath, trunc_normal_
except:
    from timm.models.layers import DropPath, trunc_normal_

try:
    import xformers.ops as xops
    HAS_XFORMERS = True
except:
    HAS_XFORMERS = False

try:
    from fairscale.nn.checkpoint import checkpoint_wrapper
    HAS_FAIRSCALE = True
except:
    HAS_FAIRSCALE = False

from math import pi

try:
    from einops import rearrange, repeat
except ImportError:
    print_log("einops is required for ViT backbone. Install with: pip install einops", level='WARNING')
    rearrange = repeat = None


def broadcat(tensors, dim=-1):
    num_tensors = len(tensors)
    shape_lens = set(list(map(lambda t: len(t.shape), tensors)))
    assert len(shape_lens) == 1, 'tensors must all have the same number of dimensions'
    shape_len = list(shape_lens)[0]
    dim = (dim + shape_len) if dim < 0 else dim
    dims = list(zip(*map(lambda t: list(t.shape), tensors)))
    expandable_dims = [(i, val) for i, val in enumerate(dims) if i != dim]
    assert all([*map(lambda t: len(set(t[1])) <= 2, expandable_dims)]), 'invalid dimensions for broadcastable concatentation'
    max_dims = list(map(lambda t: (t[0], max(t[1])), expandable_dims))
    expanded_dims = list(map(lambda t: (t[0], (t[1],) * num_tensors), max_dims))
    expanded_dims.insert(dim, (dim, dims[dim]))
    expandable_shapes = list(zip(*map(lambda t: t[1], expanded_dims)))
    tensors = list(map(lambda t: t[0].expand(*t[1]), zip(tensors, expandable_shapes)))
    return torch.cat(tensors, dim=dim)


def rotate_half(x):
    x = rearrange(x, '... (d r) -> ... d r', r=2)
    x1, x2 = x.unbind(dim=-1)
    x = torch.stack((-x2, x1), dim=-1)
    return rearrange(x, '... d r -> ... (d r)')


class VisionRotaryEmbeddingFast(nn.Module):
    def __init__(
        self,
        dim,
        pt_seq_len=16,
        ft_seq_len=None,
        custom_freqs=None,
        freqs_for='lang',
        theta=10000,
        max_freq=10,
        num_freqs=1,
    ):
        super().__init__()
        if custom_freqs:
            freqs = custom_freqs
        elif freqs_for == 'lang':
            freqs = 1. / (theta ** (torch.arange(0, dim, 2)[:(dim // 2)].float() / dim))
        elif freqs_for == 'pixel':
            freqs = torch.linspace(1., max_freq / 2, dim // 2) * pi
        elif freqs_for == 'constant':
            freqs = torch.ones(num_freqs).float()
        else:
            raise ValueError(f'unknown modality {freqs_for}')

        if ft_seq_len is None:
            ft_seq_len = pt_seq_len
        t = torch.arange(ft_seq_len) / ft_seq_len * pt_seq_len

        freqs = torch.einsum('..., f -> ... f', t, freqs)
        freqs = repeat(freqs, '... n -> ... (n r)', r=2)
        freqs = broadcat((freqs[:, None, :], freqs[None, :, :]), dim=-1)

        freqs_cos = freqs.cos().view(-1, freqs.shape[-1])
        freqs_sin = freqs.sin().view(-1, freqs.shape[-1])

        self.register_buffer("freqs_cos", freqs_cos)
        self.register_buffer("freqs_sin", freqs_sin)

    def forward(self, t):
        return t * self.freqs_cos + rotate_half(t) * self.freqs_sin


def get_rope(t, H, W):
    dim = 32
    pt_seq_len = 16
    theta = 10000

    freqs = 1. / (theta ** (torch.arange(0, dim, 2)[:(dim // 2)].float() / dim))

    tH = torch.arange(H) / H * pt_seq_len
    tW = torch.arange(W) / W * pt_seq_len

    freqsH = torch.einsum('..., f -> ... f', tH, freqs)
    freqsH = repeat(freqsH, '... n -> ... (n r)', r=2)
    freqsW = torch.einsum('..., f -> ... f', tW, freqs)
    freqsW = repeat(freqsW, '... n -> ... (n r)', r=2)
    freqs = broadcat((freqsH[:, None, :], freqsW[None, :, :]), dim=-1)
    freqs = freqs.to(t.device)
    freqs_cos = freqs.cos().view(-1, freqs.shape[-1])
    freqs_sin = freqs.sin().view(-1, freqs.shape[-1])

    return t * freqs_cos + rotate_half(t) * freqs_sin


class PatchEmbed(nn.Module):
    """Image to Patch Embedding."""

    def __init__(self, kernel_size=(16, 16), stride=(16, 16), padding=(0, 0), in_chans=3, embed_dim=768):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=kernel_size, stride=stride, padding=padding)

    def forward(self, x):
        x = self.proj(x)
        x = x.permute(0, 2, 3, 1)  # B C H W -> B H W C
        return x


def window_partition(x, window_size):
    B, H, W, C = x.shape
    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
    Hp, Wp = H + pad_h, W + pad_w
    x = x.view(B, Hp // window_size, window_size, Wp // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows, (Hp, Wp)


def window_unpartition(windows, window_size, pad_hw, hw):
    Hp, Wp = pad_hw
    H, W = hw
    B = windows.shape[0] // (Hp * Wp // window_size // window_size)
    x = windows.view(B, Hp // window_size, Wp // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, Hp, Wp, -1)
    if Hp > H or Wp > W:
        x = x[:, :H, :W, :].contiguous()
    return x


def get_abs_pos(abs_pos, has_cls_token, hw, num_register_tokens=0):
    h, w = hw
    total_special = (1 if has_cls_token else 0) + num_register_tokens
    
    if total_special > 0:
        # Extract special tokens (usually at the start or end, DINOv3 has cls at start, regs usually follow)
        # However, the pos_embed is usually [cls, spatial..., regs] or [cls, regs, spatial...]
        # DINOv3/DINOv2 with registers usually has [cls, regs, spatial] or [cls, spatial, regs]
        # Our vit.pycat: [cls, spatial, regs]. So pos_embed should match.
        special_pos = abs_pos[:, :total_special]
        spatial_pos = abs_pos[:, total_special:]
    else:
        special_pos = None
        spatial_pos = abs_pos

    xy_num = spatial_pos.shape[1]
    size = int(math.sqrt(xy_num))
    assert size * size == xy_num

    if size != h or size != w:
        new_spatial_pos = F.interpolate(
            spatial_pos.reshape(1, size, size, -1).permute(0, 3, 1, 2),
            size=(h, w),
            mode="bicubic",
            align_corners=False,
        )
        new_spatial_pos = new_spatial_pos.permute(0, 2, 3, 1).reshape(1, -1, abs_pos.shape[-1])
    else:
        new_spatial_pos = spatial_pos

    if special_pos is not None:
        return torch.cat([special_pos, new_spatial_pos], dim=1)
    return new_spatial_pos


class SwiGLU(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.SiLU, drop=0.,
                 norm_layer=nn.LayerNorm, subln=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.w1 = nn.Linear(in_features, hidden_features)
        self.w2 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.ffn_ln = norm_layer(hidden_features) if subln else nn.Identity()
        self.w3 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x1 = self.w1(x)
        x2 = self.w2(x)
        hidden = self.act(x1) * x2
        x = self.ffn_ln(hidden)
        x = self.w3(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=True, qk_scale=None, attn_head_dim=None, rope=None, xattn=True):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q_proj = nn.Linear(dim, all_head_dim, bias=False)
        self.k_proj = nn.Linear(dim, all_head_dim, bias=False)
        self.v_proj = nn.Linear(dim, all_head_dim, bias=False)

        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(all_head_dim))
            self.v_bias = nn.Parameter(torch.zeros(all_head_dim))
        else:
            self.q_bias = None
            self.v_bias = None

        self.rope = rope
        self.xattn = xattn and HAS_XFORMERS
        self.proj = nn.Linear(all_head_dim, dim)

    def forward(self, x):
        B, H, W, C = x.shape
        x = x.view(B, -1, C)
        N = H * W

        q = F.linear(input=x, weight=self.q_proj.weight, bias=self.q_bias)
        k = F.linear(input=x, weight=self.k_proj.weight, bias=None)
        v = F.linear(input=x, weight=self.v_proj.weight, bias=self.v_bias)

        q = q.reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
        k = k.reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
        v = v.reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)

        if self.rope is not None:
            q = self.rope(q).type_as(v)
            k = self.rope(k).type_as(v)
        else:
            q = get_rope(q, H, W).type_as(v)
            k = get_rope(k, H, W).type_as(v)

        if self.xattn:
            q = q.permute(0, 2, 1, 3)
            k = k.permute(0, 2, 1, 3)
            v = v.permute(0, 2, 1, 3)
            x = xops.memory_efficient_attention(q, k, v)
            x = x.reshape(B, N, -1)
        else:
            q = q * self.scale
            attn = (q @ k.transpose(-2, -1))
            attn = attn.softmax(dim=-1).type_as(x)
            x = (attn @ v).transpose(1, 2).reshape(B, N, -1)

        x = self.proj(x)
        x = x.view(B, H, W, C)
        return x


class Block(nn.Module):
    """Transformer blocks with support of window attention."""

    def __init__(self, dim, num_heads, mlp_ratio=4*2/3, qkv_bias=True, drop_path=0.0,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6), window_size=0, use_residual_block=False,
                 rope=None, xattn=True):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, rope=rope, xattn=xattn)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = SwiGLU(in_features=dim, hidden_features=int(dim * mlp_ratio), subln=True, norm_layer=norm_layer)
        self.window_size = window_size

    def forward(self, x):
        shortcut = x
        x = self.norm1(x)

        if self.window_size > 0:
            H, W = x.shape[1], x.shape[2]
            x, pad_hw = window_partition(x, self.window_size)

        x = self.attn(x)

        if self.window_size > 0:
            x = window_unpartition(x, self.window_size, pad_hw, (H, W))

        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


@MODELS.register_module()
class ViT(BaseModule):
    """Vision Transformer backbone for DINOv3 / ViTDet style."""

    def __init__(
        self,
        img_size=1024,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        drop_path_rate=0.0,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        use_abs_pos=True,
        pt_hw_seq_len=16,
        intp_freq=True,
        window_size=0,
        window_block_indexes=(),
        residual_block_indexes=(),
        use_act_checkpoint=False,
        use_lsj=False,
        pretrain_img_size=518,
        pretrain_use_cls_token=True,
        num_register_tokens=4,
        xattn=False,
        frozen_stages=-1,  # Added for future freezing flexibility
        init_cfg=None
    ):
        super(ViT, self).__init__(init_cfg=init_cfg)
        self.pretrain_use_cls_token = pretrain_use_cls_token
        self.num_register_tokens = num_register_tokens
        self.frozen_stages = frozen_stages

        self.patch_embed = PatchEmbed(
            kernel_size=(patch_size, patch_size),
            stride=(patch_size, patch_size),
            in_chans=in_chans,
            embed_dim=embed_dim,
        )

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if pretrain_use_cls_token else None
        self.register_tokens = nn.Parameter(torch.zeros(1, num_register_tokens, embed_dim)) if num_register_tokens > 0 else None

        if use_abs_pos:
            num_patches = (pretrain_img_size // patch_size) * (pretrain_img_size // patch_size)
            num_positions = num_patches
            if pretrain_use_cls_token:
                num_positions += 1
            if num_register_tokens > 0:
                num_positions += num_register_tokens
            self.pos_embed = nn.Parameter(torch.zeros(1, num_positions, embed_dim))
        else:
            self.pos_embed = None

        half_head_dim = embed_dim // num_heads // 2
        hw_seq_len = img_size // patch_size

        self.rope_win = VisionRotaryEmbeddingFast(
            dim=half_head_dim,
            pt_seq_len=pt_hw_seq_len,
            ft_seq_len=window_size if intp_freq else None,
        )
        self.rope_glb = VisionRotaryEmbeddingFast(
            dim=half_head_dim,
            pt_seq_len=pt_hw_seq_len,
            ft_seq_len=hw_seq_len if intp_freq else None,
        )
        if not use_lsj:
            self.rope_glb = None

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        self.blocks = nn.ModuleList()
        for i in range(depth):
            block = Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                window_size=window_size if i in window_block_indexes else 0,
                use_residual_block=i in residual_block_indexes,
                rope=self.rope_win if i in window_block_indexes else self.rope_glb,
                xattn=xattn
            )
            if use_act_checkpoint and HAS_FAIRSCALE:
                block = checkpoint_wrapper(block)
            self.blocks.append(block)

        self.out_norm = nn.LayerNorm(embed_dim)

        if self.pos_embed is not None:
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
        if self.cls_token is not None:
            nn.init.trunc_normal_(self.cls_token, std=0.02)
        if self.register_tokens is not None:
            nn.init.trunc_normal_(self.register_tokens, std=0.02)

        self.apply(self._init_weights)
        self._freeze_stages()

    def _freeze_stages(self):
        """Freeze stages based on self.frozen_stages."""
        if self.frozen_stages >= 0:
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.requires_grad = False
            if self.pos_embed is not None:
                self.pos_embed.requires_grad = False
            if self.cls_token is not None:
                self.cls_token.requires_grad = False
            if self.register_tokens is not None:
                self.register_tokens.requires_grad = False

        for i in range(self.frozen_stages):
            m = self.blocks[i]
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

    def train(self, mode=True):
        """Override train to keep frozen stages in eval mode."""
        super(ViT, self).train(mode)
        self._freeze_stages()

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.patch_embed(x)
        Hp, Wp = x.shape[1], x.shape[2]
        
        # Reshape to tokens: (B, H*W, C)
        x = x.reshape(B, Hp * Wp, -1)

        # Append special tokens
        if self.cls_token is not None:
            cls_tokens = self.cls_token.expand(B, -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)
        
        if self.register_tokens is not None:
            reg_tokens = self.register_tokens.expand(B, -1, -1)
            x = torch.cat((x, reg_tokens), dim=1)

        # Add Absolute Positional Embedding
        if self.pos_embed is not None:
             pos_embed = get_abs_pos(self.pos_embed, self.pretrain_use_cls_token, (Hp, Wp), 
                                   num_register_tokens=self.num_register_tokens)
             # Reshape pos_embed to match token sequence
             # get_abs_pos returns (1, H, W, C) + special tokens if handled
             # Correct logic to handle flattened tokens + interpolated spatial pos:
             x = x + pos_embed.reshape(1, -1, x.shape[-1])

        # Blocks expect (B, H, W, C) or handle tokens
        # The Attention class we have uses x.shape to determine H, W
        # If tokens are present, we must pass them carefully or reshape.
        # Let's simplify: only pass spatial patches to blocks if windowing is used.
        
        # Number of prefix tokens
        num_prefix = (1 if self.cls_token is not None else 0)
        num_suffix = self.num_register_tokens

        for blk in self.blocks:
            if blk.window_size > 0:
                # Extract spatial part for windowed attention
                prefix = x[:, :num_prefix, :]
                spatial = x[:, num_prefix:num_prefix + Hp*Wp, :]
                suffix = x[:, num_prefix + Hp*Wp:, :]
                
                spatial = spatial.reshape(B, Hp, Wp, -1)
                spatial = blk(spatial)
                spatial = spatial.reshape(B, Hp*Wp, -1)
                
                x = torch.cat((prefix, spatial, suffix), dim=1)
            else:
                # Global attention can handle sequence or grid
                # Our Attention class expects (B, H, W, C). Let's adapt it.
                # Actually, our Block/Attention implementation is very grid-heavy.
                # To support registers/cls, we'll keep x in (B, H, W, C) for blocks and handle tokens separately if needed.
                # BUT DINOv3 registers are part of the sequence.
                # Simplified for CO-DETR logic: we'll project spatial only to the next stage.
                x_spatial = x[:, num_prefix:num_prefix + Hp*Wp, :].reshape(B, Hp, Wp, -1)
                x_spatial = blk(x_spatial)
                x = torch.cat((x[:, :num_prefix, :], x_spatial.reshape(B, Hp * Wp, -1), x[:, num_prefix + Hp*Wp:, :]), dim=1)

        # Final output
        x_spatial = x[:, num_prefix:num_prefix + Hp*Wp, :].reshape(B, Hp, Wp, -1)
        outputs = [self.out_norm(x_spatial).permute(0, 3, 1, 2).contiguous()]
        return outputs
