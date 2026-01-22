# Copyright (c) OpenMMLab. All rights reserved.
# Rewritten to match DINOv3 checkpoint architecture
import logging
import math
from functools import partial
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from mmengine.model import BaseModule
from mmengine.runner import load_checkpoint
from mmengine.logging import print_log
from mmdet.registry import MODELS

try:
    from timm.layers import DropPath, trunc_normal_
except:
    from timm.models.layers import DropPath, trunc_normal_

try:
    import torchao
    from torchao.quantization import float8_weight_only, quantize_
    HAS_TORCHAO = True
except ImportError:
    HAS_TORCHAO = False

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


class PatchEmbed(nn.Module):
    """Image to Patch Embedding."""

    def __init__(self, kernel_size=(16, 16), stride=(16, 16), padding=(0, 0), in_chans=3, embed_dim=768):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=kernel_size, stride=stride, padding=padding)

    def forward(self, x):
        x = self.proj(x)
        x = x.permute(0, 2, 3, 1)  # B C H W -> B H W C
        return x


class RoPE2D(nn.Module):
    """2D Rotary Position Embedding matching DINOv3 implementation.
    
    DINOv3 uses a periods-based 2D RoPE with 16 periods for ViT-B.
    The periods buffer is loaded from the checkpoint.
    """
    
    def __init__(self, num_periods=16):
        super().__init__()
        # Periods will be loaded from checkpoint - DINOv3 uses 16
        self.register_buffer("periods", torch.ones(num_periods))
        
    def forward(self, x, h, w):
        """Apply 2D RoPE to input tensor.
        
        Args:
            x: (B, num_heads, N, head_dim) tensor
            h, w: spatial dimensions
        Returns:
            Rotated tensor of same shape
        """
        B, num_heads, N, head_dim = x.shape
        device = x.device
        dtype = x.dtype
        
        # Use loaded periods from checkpoint
        periods = self.periods.float()  # computing angles in float32 for stability
        num_periods = periods.shape[0]
        rope_dim = num_periods * 4
        
        # Generate 2D position grid
        pos_h = torch.arange(h, device=device, dtype=torch.float32)
        pos_w = torch.arange(w, device=device, dtype=torch.float32)
        grid_h, grid_w = torch.meshgrid(pos_h, pos_w, indexing='ij')
        
        # Compute angles using periods
        angles_h = grid_h.flatten()[:, None] / periods[None, :]
        angles_w = grid_w.flatten()[:, None] / periods[None, :]
        
        # Interleave h and w angles, then compute sin/cos
        angles = torch.stack([angles_h, angles_w], dim=-1)
        angles = angles.reshape(N, -1)
        
        cos_angles = angles.cos()
        sin_angles = angles.sin()
        
        # Only apply RoPE to the first rope_dim dimensions
        if rope_dim > head_dim:
            rope_dim = head_dim
        
        rope_dim_half = rope_dim // 2
        
        cos_angles = cos_angles[:, :rope_dim_half].unsqueeze(0).unsqueeze(0).to(dtype)
        sin_angles = sin_angles[:, :rope_dim_half].unsqueeze(0).unsqueeze(0).to(dtype)
        
        # Split x into RoPE part and passthrough part
        x_rope = x[..., :rope_dim]
        x_pass = x[..., rope_dim:]
        
        # Split RoPE part in half for rotation
        x1, x2 = x_rope[..., :rope_dim_half], x_rope[..., rope_dim_half:]
        
        # Apply rotation (in original dtype)
        x_rotated = torch.cat([
            x1 * cos_angles - x2 * sin_angles,
            x1 * sin_angles + x2 * cos_angles,
            x_pass
        ], dim=-1)
        
        return x_rotated


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


class Attention(nn.Module):
    """Multi-head attention with fused QKV projection matching DINOv3."""

    def __init__(self, dim, num_heads=8, qkv_bias=True, xattn=True):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        # Fused QKV projection (matches DINOv3 checkpoint)
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        
        self.xattn = xattn and HAS_XFORMERS

    def forward(self, x, rope=None, h=None, w=None):
        """
        Args:
            x: (B, N, C) tensor
            rope: optional RoPE module
            h, w: spatial dimensions for RoPE
        """
        B, N, C = x.shape
        
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # Each: (B, num_heads, N, head_dim)
        
        # Apply RoPE if provided
        if rope is not None and h is not None and w is not None:
            # Calculate number of extra tokens
            num_spatial = h * w
            # cls token is usually at index 0
            # spatial tokens are from index 1 to 1 + num_spatial
            # storage tokens are after spatial tokens
            
            # Extract spatial part for RoPE
            # Assuming [cls, spatial..., storage...] layout from ViT.forward
            q_prefix = q[:, :, :1]
            q_spatial = q[:, :, 1:1+num_spatial]
            q_suffix = q[:, :, 1+num_spatial:]
            
            k_prefix = k[:, :, :1]
            k_spatial = k[:, :, 1:1+num_spatial]
            k_suffix = k[:, :, 1+num_spatial:]

            # Apply RoPE to spatial tokens only
            q_spatial = rope(q_spatial, h, w)
            k_spatial = rope(k_spatial, h, w)
            
            # Reconstruct
            q = torch.cat([q_prefix, q_spatial, q_suffix], dim=2)
            k = torch.cat([k_prefix, k_spatial, k_suffix], dim=2)

        if self.xattn:
            # xformers expects (B, N, num_heads, head_dim)
            q = q.permute(0, 2, 1, 3)
            k = k.permute(0, 2, 1, 3)
            v = v.permute(0, 2, 1, 3)
            # Ensure contiguous for xformers
            q = q.contiguous()
            k = k.contiguous()
            v = v.contiguous()
            x = xops.memory_efficient_attention(q, k, v)
            x = x.reshape(B, N, C)
        else:
            q = q * self.scale
            # (B, num_heads, N, head_dim) @ (B, num_heads, head_dim, N) -> (B, num_heads, N, N)
            attn = (q @ k.transpose(-2, -1))
            
            # Clamp attention scores for stability
            attn = torch.clamp(attn, min=-50, max=50)
            
            # Perform softmax in float32 for numerical stability
            attn = attn.float().softmax(dim=-1).type_as(q)
            x = (attn @ v).transpose(1, 2).reshape(B, N, C)

        x = self.proj(x)
        return x


class Mlp(nn.Module):
    """Standard MLP matching DINOv3 checkpoint (not SwiGLU)."""
    
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class LayerScale(nn.Module):
    """Layer Scale from CaiT/DeiT-III."""
    
    def __init__(self, dim, init_values=1e-5):
        super().__init__()
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        out = x * self.gamma
        return out


class Block(nn.Module):
    """Transformer block matching DINOv3 architecture."""

    def __init__(self, dim, num_heads, mlp_ratio=4.0, qkv_bias=True, drop_path=0.0,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6), window_size=0, 
                 init_values=1e-5, xattn=True):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, xattn=xattn)
        self.ls1 = LayerScale(dim, init_values=init_values)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim)
        self.ls2 = LayerScale(dim, init_values=init_values)
        
        self.window_size = window_size

    def forward(self, x, rope=None, h=None, w=None):
        """
        Args:
            x: (B, N, C) tensor with cls + spatial + register tokens
            rope: optional RoPE module
            h, w: spatial dimensions
        """
        if self.window_size > 0:
            # Extract parts
            num_prefix = 1 # cls
            num_spatial = h * w
            
            prefix = x[:, :num_prefix, :]
            spatial = x[:, num_prefix:num_prefix + num_spatial, :]
            suffix = x[:, num_prefix + num_spatial:, :]
            
            # Reshape spatial for windowing
            spatial = spatial.reshape(x.shape[0], h, w, x.shape[-1])
            spatial, pad_hw = window_partition(spatial, self.window_size)
            
            # Apply attention on windows (RoPE usually not applied in small windows or needs adjustment)
            # For simplicity with CO-DETR, we'll skip RoPE in windowed blocks if it's too complex, 
            # but let's try to keep it if h/w are passed as window_size
            B_win = spatial.shape[0]
            spatial = spatial.reshape(B_win, -1, spatial.shape[-1])
            
            # Attention within window
            # Note: rope is skipped here because spatial positions change within windows
            spatial = self.attn(self.norm1(spatial)) 
            
            # Unpartition
            spatial = window_unpartition(spatial, self.window_size, pad_hw, (h, w))
            spatial = spatial.reshape(x.shape[0], num_spatial, -1)
            
            # Reconstruct and apply MLP
            x_spatial = torch.cat([prefix, spatial, suffix], dim=1)
            x_attn = x + self.drop_path(self.ls1(x_spatial))
            x = x_attn + self.drop_path(self.ls2(self.mlp(self.norm2(x_attn))))
        else:
            # Global attention
            x_attn = x + self.drop_path(self.ls1(self.attn(self.norm1(x), rope=rope, h=h, w=w)))
            x = x_attn + self.drop_path(self.ls2(self.mlp(self.norm2(x_attn))))
            
        return x


@MODELS.register_module()
class ViT(BaseModule):
    """Vision Transformer backbone matching DINOv3 architecture."""

    def __init__(
        self,
        img_size=1024,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_path_rate=0.0,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        pretrain_img_size=518,
        pretrain_use_cls_token=True,
        num_register_tokens=4,
        xattn=False,
        init_values=1e-5,
        use_rope=True,
        window_size=0,
        window_block_indexes=(),
        use_act_checkpoint=False,
        frozen_stages=-1,
        init_cfg=None
    ):
        super(ViT, self).__init__(init_cfg=init_cfg)
        self.pretrain_use_cls_token = pretrain_use_cls_token
        self.num_register_tokens = num_register_tokens
        self.frozen_stages = frozen_stages
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        self.patch_embed = PatchEmbed(
            kernel_size=(patch_size, patch_size),
            stride=(patch_size, patch_size),
            in_chans=in_chans,
            embed_dim=embed_dim,
        )

        # Special tokens
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if pretrain_use_cls_token else None
        # Named storage_tokens to match DINOv3 checkpoint (will be mapped in load)
        self.storage_tokens = nn.Parameter(torch.zeros(1, num_register_tokens, embed_dim)) if num_register_tokens > 0 else None

        # RoPE embedding - DINOv3 uses 16 periods
        self.use_rope = use_rope
        if use_rope:
            self.rope_embed = RoPE2D(num_periods=16)
        else:
            self.rope_embed = None

        # Stochastic depth decay
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
                init_values=init_values,
                xattn=xattn
            )
            if use_act_checkpoint and HAS_FAIRSCALE:
                block = checkpoint_wrapper(block)
            self.blocks.append(block)

        # Output norm
        self.norm = norm_layer(embed_dim)

        # Initialize
        if self.cls_token is not None:
            nn.init.trunc_normal_(self.cls_token, std=0.02)
        if self.storage_tokens is not None:
            nn.init.trunc_normal_(self.storage_tokens, std=0.02)

        self.apply(self._init_weights)
        self._freeze_stages()
        
        # Apply torch.compile to blocks if possible
        if hasattr(torch, 'compile'):
            try:
                self.blocks = torch.compile(self.blocks, mode='default')
                print_log("Applied torch.compile to ViT blocks", level='INFO')
            except Exception as e:
                print_log(f"Failed to apply torch.compile: {e}", level='WARNING')

    def init_weights(self):
        """Initialize weights and apply FP8 optimization after loading."""
        super().init_weights()
        # After weights are loaded (from init_cfg), apply FP8 quantization to the backbone
        self.apply_fp8_optimization()

    def apply_fp8_optimization(self):
        """Apply torchao FP8 quantization to frozen linear layers."""
        if not HAS_TORCHAO:
            print_log("torchao not found, skipping FP8 optimization", level='WARNING')
            return
            
        print_log("Applying float8_weight_only quantization to ViT backbone...", level='INFO')
        # We target linear layers in blocks
        # float8_weight_only is suitable for frozen backbones to save memory and speed up matmuls
        try:
            quantize_(self, float8_weight_only())
            print_log("Successfully applied FP8 quantization", level='INFO')
        except Exception as e:
            print_log(f"FP8 quantization failed: {e}", level='ERROR')

    def _freeze_stages(self):
        """Freeze stages based on self.frozen_stages."""
        if self.frozen_stages >= 0:
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.requires_grad = False
            if self.cls_token is not None:
                self.cls_token.requires_grad = False
            if self.storage_tokens is not None:
                self.storage_tokens.requires_grad = False

        for i in range(self.frozen_stages):
            if i < len(self.blocks):
                m = self.blocks[i]
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False
        
        if self.frozen_stages >= len(self.blocks):
            self.norm.eval()
            for param in self.norm.parameters():
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
        x = self.patch_embed(x)  # (B, Hp, Wp, C)
        Hp, Wp = x.shape[1], x.shape[2]
        
        # Flatten spatial dims: (B, Hp*Wp, C)
        x = x.reshape(B, Hp * Wp, -1)

        # Prepend cls token
        if self.cls_token is not None:
            cls_tokens = self.cls_token.expand(B, -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)
        
        # Append register/storage tokens
        if self.storage_tokens is not None:
            reg_tokens = self.storage_tokens.expand(B, -1, -1)
            x = torch.cat((x, reg_tokens), dim=1)

        # Pass through transformer blocks
        for blk in self.blocks:
            x = blk(x, rope=self.rope_embed, h=Hp, w=Wp)

        # Apply final norm
        x = self.norm(x)

        # Extract spatial features (remove cls and register tokens)
        num_prefix = 1 if self.cls_token is not None else 0
        num_suffix = self.num_register_tokens
        
        x_spatial = x[:, num_prefix:num_prefix + Hp*Wp, :]
        x_spatial = x_spatial.reshape(B, Hp, Wp, -1)
        
        # Output as (B, C, H, W) for detection head
        outputs = [x_spatial.permute(0, 3, 1, 2).contiguous()]
        return outputs
