# -*- coding: utf-8 -*-
"""
================================================
@author: Jaron
@time: 2024/12/25 12:17:44
@email: fjjth98@163.com
@description:
Modified from transformers.models.qwen2_vl.modeling_qwen2_vl.py, LICENSE:
Copyright 2024 The Qwen team, Alibaba Group and the HuggingFace Inc. team. All rights reserved.

This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
and OPT implementations in this library. It has been modified from its
original forms to accommodate minor architectural differences compared
to GPT-NeoX and OPT used by the Meta AI team that trained the model.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
================================================
"""
import math

import torch
from torch import nn
from torch.nn import functional as F
from transformers import PreTrainedModel
from transformers.activations import ACT2FN

try:
    from flash_attn import flash_attn_varlen_func
except:
    pass



from .configuration_qwen2vl_navit import Qwen2VLNaViTConfig


def apply_rope(x: torch.Tensor, freqs: torch.Tensor, unsqueeze_dim: int = 0) -> torch.Tensor:
    """
    Args:
        x (torch.Tensor): (H, L, C) / (L, H, C)
        freqs (torch.Tensor): (L, C // 2)
        unsqueeze_dim (int): the dimension of unsqueeze, 0 for (H, L, C), 1 for (L, H, C)

    Returns:
        torch.Tensor: (H, L, C)
    """
    x1, x2 = x.to(torch.float32).chunk(2, dim=-1)
    freqs = freqs.to(torch.float32).unsqueeze(dim=unsqueeze_dim)
    cos, sin = freqs.cos(), freqs.sin()
    output = torch.cat((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1).to(x.dtype)
    return output


class VisionRotaryEmbedding(nn.Module):
    def __init__(self, dim: int, theta: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seqlen: int) -> torch.Tensor:
        """
        Args:
            seqlen (int): _description_

        Returns:
            torch.Tensor: (seqlen, dim // 2)
        """
        seq = torch.arange(seqlen, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(seq, self.inv_freq)
        return freqs


class VisionMlp(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, hidden_act: str):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = ACT2FN[hidden_act]
        self.fc2 = nn.Linear(hidden_dim, dim)

    def forward(self, x) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class VisionAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 16):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(
        self, hidden_states: torch.Tensor, cu_seqlens: torch.Tensor, rotary_pos_emb: torch.Tensor
    ) -> torch.Tensor:
        seq_length = hidden_states.size(0)
        q, k, v = self.qkv(hidden_states).view(seq_length, 3, self.num_heads, -1).permute(1, 2, 0, 3).unbind(0)     # (H, L, C)
        q, k = apply_rope(q, rotary_pos_emb), apply_rope(k, rotary_pos_emb)

        attn_output = []
        for i in range(1, len(cu_seqlens)):
            attn_weights = torch.matmul(
                q[:, cu_seqlens[i-1]:cu_seqlens[i]],                # (H, l, C)
                k[:, cu_seqlens[i-1]:cu_seqlens[i]].transpose(1, 2) # (H, C, l)
            ) / math.sqrt(self.head_dim)        # (H, l, l)
            attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
            attn_output.append(torch.bmm(
                attn_weights, v[:, cu_seqlens[i-1]:cu_seqlens[i]]
            ).transpose(0, 1))      # (l, H, C)
        attn_output = torch.cat(attn_output, dim=0).flatten(1)     # (L, C')
        attn_output = self.proj(attn_output)

        return attn_output


class VisionFlashAttention2(VisionAttention):

    def forward(
        self, hidden_states: torch.Tensor, cu_seqlens: torch.Tensor, rotary_pos_emb: torch.Tensor
    ) -> torch.Tensor:
        seq_length = hidden_states.size(0)
        q, k, v = self.qkv(hidden_states).view(seq_length, 3, self.num_heads, -1).unbind(1)     # (L, H, C)
        q, k = apply_rope(q, rotary_pos_emb, 1), apply_rope(k, rotary_pos_emb, 1)

        max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()
        attn_output = flash_attn_varlen_func(
            q, k, v,
            cu_seqlens, cu_seqlens,
            max_seqlen, max_seqlen
        ).flatten(1)
        attn_output = self.proj(attn_output)

        return attn_output


class VisionSdpaAttention(VisionAttention):

    def forward(
        self, hidden_states: torch.Tensor, cu_seqlens: torch.Tensor, rotary_pos_emb: torch.Tensor
    ) -> torch.Tensor:
        seq_length = hidden_states.size(0)
        q, k, v = self.qkv(hidden_states).view(seq_length, 3, self.num_heads, -1).permute(1, 2, 0, 3).unbind(0)     # (H, L, C)
        q, k = apply_rope(q, rotary_pos_emb), apply_rope(k, rotary_pos_emb)

        if cu_seqlens.size(0) == 2:
            attn_output = F.scaled_dot_product_attention(q, k, v)   # (H, L, C)
        else:
            seqlens = cu_seqlens.diff().tolist()
            q = torch.nested.as_nested_tensor(list(q.split(seqlens, dim=1)))
            k = torch.nested.as_nested_tensor(list(k.split(seqlens, dim=1)))
            v = torch.nested.as_nested_tensor(list(v.split(seqlens, dim=1)))
            attn_output = F.scaled_dot_product_attention(q, k, v)
            attn_output = torch.cat(attn_output.unbind(), dim=1)
        attn_output = self.proj(attn_output.transpose(0, 1).flatten(1))

        return attn_output


QWEN2_VL_NAVIT_ATTENTION_CLASSES = {
    "eager": VisionAttention,
    "flash_attention_2": VisionFlashAttention2,
    "sdpa": VisionSdpaAttention,
}


class Qwen2VLVisionBlock(nn.Module):
    def __init__(self, config, attn_implementation: str = "sdpa"):
        super().__init__()
        self.norm1 = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.norm2 = nn.LayerNorm(config.hidden_size, eps=1e-6)
        mlp_hidden_dim = int(config.hidden_size * config.mlp_ratio)

        self.attn = QWEN2_VL_NAVIT_ATTENTION_CLASSES[attn_implementation](
            config.hidden_size, num_heads=config.num_heads
        )
        self.mlp = VisionMlp(dim=config.hidden_size, hidden_dim=mlp_hidden_dim, hidden_act=config.hidden_act)

    def forward(self, hidden_states, cu_seqlens, rotary_pos_emb) -> torch.Tensor:
        hidden_states = hidden_states + self.attn(
            self.norm1(hidden_states), cu_seqlens=cu_seqlens, rotary_pos_emb=rotary_pos_emb
        )
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
        return hidden_states


class Qwen2VLNaViT(PreTrainedModel):
    config_class = Qwen2VLNaViTConfig
    _no_split_modules = ["Qwen2VLVisionBlock"]
    supports_gradient_checkpointing = True
    _supports_flash_attn_2 = True
    _supports_sdpa = True

    def __init__(self, config):
        super().__init__(config)
        self.gradient_checkpointing = False

        self.patch_embed = nn.Conv2d(
            in_channels=config.in_channels,
            out_channels=config.hidden_size,
            kernel_size=config.patch_size,
            stride=config.patch_size,
            bias=False
        )
        self.rotary_pos_emb = VisionRotaryEmbedding(config.hidden_size // config.num_heads // 2)
        self.blocks = nn.ModuleList(
            [Qwen2VLVisionBlock(config, config._attn_implementation) for _ in range(config.depth)]
        )

    def rot_pos_emb(self, grid_sizes: torch.LongTensor) -> torch.Tensor:
        """
        Args:
            grid_sizes (torch.LongTensor): (B, 2)

        Returns:
            torch.Tensor: (L, C // 2)
        """
        pos_ids = torch.cat(
            [torch.cartesian_prod(torch.arange(h), torch.arange(w)) for h, w in grid_sizes]
        , dim=0)    # (L, 2)
        rotary_pos_emb = self.rotary_pos_emb(grid_sizes.max())[pos_ids].flatten(1)
        return rotary_pos_emb

    @classmethod
    def _check_and_enable_sdpa(cls, config: Qwen2VLNaViTConfig, hard_check_only: bool = False) -> Qwen2VLNaViTConfig:
        """rewrite this function to avoid error on torch==2.1.0"""
        return config

    def forward(self, hidden_states: torch.Tensor, grid_sizes: torch.LongTensor) -> torch.Tensor:
        """
        Args:
            hidden_states (torch.Tensor): (\sum_i h_i*w_i, 3, patch_size, patch_size)
            grid_sizes (torch.LongTensor): (B, 2)

        Returns:
            torch.Tensor: (L, C)
        """
        hidden_states = self.patch_embed(hidden_states).flatten(1)
        rotary_pos_emb = self.rot_pos_emb(grid_sizes)
        cu_seqlens = F.pad((grid_sizes[:, 0] * grid_sizes[:, 1]).cumsum(dim=0, dtype=torch.int32), (1, 0), value=0)

        # debug_output = [hidden_states]
        for blk in self.blocks:
            if self.gradient_checkpointing and self.training:
                hidden_states = self._gradient_checkpointing_func(
                    blk.__call__, hidden_states, cu_seqlens, rotary_pos_emb
                )
            else:
                hidden_states = blk(hidden_states, cu_seqlens=cu_seqlens, rotary_pos_emb=rotary_pos_emb)
            # debug_output.append(hidden_states)

        return hidden_states
