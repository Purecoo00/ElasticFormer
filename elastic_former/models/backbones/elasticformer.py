# Copyright (c) 2026 ElasticFormer authors.
# Global/local attention follows liwenxi/SparseFormer and OpenMMLab.
# SPDX-License-Identifier: Apache-2.0
"""ElasticFormer backbone with stage-wise foreground-aware window selection."""

import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from mmcv.cnn.bricks.drop import DropPath
from mmcv.cnn.bricks.transformer import FFN
from mmdet.models.backbones.swin import SwinTransformer, WindowMSA
from mmdet.registry import MODELS


def partition(x, window_size):
    """Split a padded BHWC map into (batch, windows, tokens, channels)."""
    b, h, w, c = x.shape
    s = window_size
    return x.reshape(b, h // s, s, w // s, s, c).permute(0, 1, 3, 2, 4, 5).reshape(b, -1, s * s, c)


def reverse(windows, height, width, window_size):
    """Put window tokens back in their spatial order."""
    b, _, _, c = windows.shape
    s = window_size
    return (
        windows.reshape(b, height // s, width // s, s, s, c)
        .permute(0, 1, 3, 2, 4, 5)
        .reshape(b, height, width, c)
    )


class GlobalAttention(nn.Module):
    """Self-attention over one pooled token per window."""

    def __init__(self, dim, num_heads, qkv_bias, qk_scale, drop, attn_drop, drop_path):
        super().__init__()
        self.num_heads = num_heads
        self.scale = qk_scale or (dim // num_heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(drop)
        self.out_drop = DropPath(drop_path) if drop_path else nn.Identity()

    def forward(self, x):
        b, n, c = x.shape
        q, k, v = (
            self.qkv(x)
            .reshape(b, n, 3, self.num_heads, c // self.num_heads)
            .permute(2, 0, 3, 1, 4)
            .unbind(0)
        )
        weights = self.attn_drop(((q @ k.transpose(-2, -1)) * self.scale).softmax(-1))
        x = (weights @ v).transpose(1, 2).reshape(b, n, c)
        return self.out_drop(self.proj_drop(self.proj(x)))


class GlobalBlock(nn.Module):
    def __init__(
        self, dim, num_heads, mlp_ratio, qkv_bias, qk_scale, drop, attn_drop, drop_path, with_cp
    ):
        super().__init__()
        self.norm1, self.norm2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.attn = GlobalAttention(dim, num_heads, qkv_bias, qk_scale, drop, attn_drop, drop_path)
        self.ffn = FFN(
            dim,
            int(dim * mlp_ratio),
            ffn_drop=drop,
            act_cfg=dict(type="GELU"),
            dropout_layer=dict(type="DropPath", drop_prob=drop_path),
            add_identity=True,
        )
        self.with_cp = with_cp

    def forward(self, x):
        def attend(tokens):
            tokens = tokens + self.attn(self.norm1(tokens))
            return self.ffn(self.norm2(tokens), identity=tokens)

        if self.with_cp and self.training:
            return checkpoint(attend, x, use_reentrant=False)
        return attend(x)


class LocalAttention(nn.Module):
    """Apply window attention only to selected windows."""

    def __init__(
        self, dim, num_heads, window_size, shift, qkv_bias, qk_scale, drop, attn_drop, drop_path
    ):
        super().__init__()
        self.window_size = window_size
        self.shift = window_size // 2 if shift else 0
        self.w_msa = WindowMSA(
            dim,
            num_heads,
            (window_size, window_size),
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop_rate=attn_drop,
            proj_drop_rate=drop,
        )
        self.drop = DropPath(drop_path) if drop_path else nn.Identity()

    def forward(self, x, hw_shape, selections):
        b, _, c = x.shape
        h, w = hw_shape
        s, shift = self.window_size, self.shift
        spatial = F.pad(x.reshape(b, h, w, c), (0, 0, 0, (-w) % s, 0, (-h) % s))
        hp, wp = spatial.shape[1:3]
        valid = torch.zeros(b, hp, wp, 1, device=x.device, dtype=torch.bool)
        valid[:, :h, :w] = True
        regions = torch.zeros(1, hp, wp, 1, device=x.device, dtype=torch.long)
        if shift:
            # Prevent cyclic shifts from connecting opposite image edges.
            label = 0
            for hs in (slice(0, -s), slice(-s, -shift), slice(-shift, None)):
                for ws in (slice(0, -s), slice(-s, -shift), slice(-shift, None)):
                    regions[:, hs, ws] = label
                    label += 1
            spatial = torch.roll(spatial, (-shift, -shift), (1, 2))
            valid = torch.roll(valid, (-shift, -shift), (1, 2))
        windows = partition(spatial, s)
        masks = partition(valid, s).squeeze(-1)
        labels = partition(regions, s).squeeze(-1)[0]
        # Unselected windows contribute no residual update.
        updates = torch.zeros_like(windows)
        for image, indices in enumerate(selections):
            tokens = windows[image, indices]
            keep = masks[image, indices]
            region = labels[indices]
            allowed = keep[:, None, :] & (region[:, :, None] == region[:, None, :])
            mask = torch.zeros_like(allowed, dtype=x.dtype).masked_fill(~allowed, -100.0)
            updates[image, indices] = self.w_msa(tokens, mask) * keep[..., None]
        spatial = reverse(updates, hp, wp, s)
        if shift:
            spatial = torch.roll(spatial, (shift, shift), (1, 2))
        return self.drop(spatial[:, :h, :w].reshape(b, h * w, c))


class ElasticBlock(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        window_size,
        shift,
        mlp_ratio,
        qkv_bias,
        qk_scale,
        drop,
        attn_drop,
        drop_path,
        with_cp,
    ):
        super().__init__()
        self.window_size, self.with_cp = window_size, with_cp
        self.norm1, self.norm2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.attn = LocalAttention(
            dim, num_heads, window_size, shift, qkv_bias, qk_scale, drop, attn_drop, drop_path
        )
        self.ffn = FFN(
            dim,
            int(dim * mlp_ratio),
            ffn_drop=drop,
            act_cfg=dict(type="GELU"),
            dropout_layer=dict(type="DropPath", drop_prob=drop_path),
            add_identity=False,
        )

    def forward(self, x, hw_shape, selections):
        def attend(tokens):
            tokens = tokens + self.attn(self.norm1(tokens), hw_shape, selections)
            b, _, c = tokens.shape
            h, w = hw_shape
            s = self.window_size
            normalized = F.pad(
                self.norm2(tokens).reshape(b, h, w, c), (0, 0, 0, (-w) % s, 0, (-h) % s)
            )
            hp, wp = normalized.shape[1:3]
            windows = partition(normalized, s)
            updates = torch.zeros_like(windows)
            # The FFN follows the same sparse window budget as local attention.
            for image, indices in enumerate(selections):
                updates[image, indices] = self.ffn(windows[image, indices])
            return tokens + reverse(updates, hp, wp, s)[:, :h, :w].reshape(b, h * w, c)

        if self.with_cp and self.training:
            return checkpoint(attend, x, use_reentrant=False)
        return attend(x)


class ElasticStage(nn.Module):
    """Global context, ElasticSelector, and sparse local blocks for one stage."""

    def __init__(
        self,
        dim,
        num_heads,
        downsample,
        pool_size,
        hidden_dim,
        drop_paths,
        window_size,
        mlp_ratio,
        qkv_bias,
        qk_scale,
        drop,
        attn_drop,
        with_cp,
    ):
        super().__init__()
        self.window_size = window_size
        self.pooling = nn.AvgPool2d(window_size, window_size)
        self.adapt_pool = nn.AdaptiveAvgPool2d(pool_size)
        self.top_k_generator = nn.Sequential(
            nn.Linear(dim * pool_size**2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )
        self.score_net = nn.Linear(window_size**2 * dim, 1)
        self.score_norm = nn.LayerNorm(dim)
        self.global_blocks = nn.ModuleList(
            [
                GlobalBlock(
                    dim, num_heads, mlp_ratio, qkv_bias, qk_scale, drop, attn_drop, path, with_cp
                )
                for path in drop_paths
            ]
        )
        self.local_blocks = nn.ModuleList(
            [
                ElasticBlock(
                    dim,
                    num_heads,
                    window_size,
                    i % 2 == 1,
                    mlp_ratio,
                    qkv_bias,
                    qk_scale,
                    drop,
                    attn_drop,
                    path,
                    with_cp,
                )
                for i, path in enumerate(drop_paths)
            ]
        )
        self.downsample = downsample
        self.use_dynamic_topk = True
        self.train_selector = True
        self.last_ratios = None

    def select_windows(self, scores, ratios):
        selections = []
        for image in range(scores.shape[0]):
            # Keep each image's budget independent, including batches above one.
            if self.training and not self.use_dynamic_topk:
                count = scores.shape[1]
            else:
                count = min(
                    scores.shape[1],
                    max(1, math.ceil(float(ratios[image].detach()) * scores.shape[1])),
                )
            selections.append(scores[image].topk(count).indices)
        return selections

    def forward(self, x, hw_shape):
        b, _, c = x.shape
        h, w = hw_shape
        s = self.window_size
        spatial = F.pad(x.reshape(b, h, w, c), (0, 0, 0, (-w) % s, 0, (-h) % s))
        hp, wp = spatial.shape[1:3]
        pooled = self.pooling(spatial.permute(0, 3, 1, 2))
        mean = F.interpolate(pooled, size=(hp, wp), mode="bilinear", align_corners=False).permute(
            0, 2, 3, 1
        )
        residual = spatial - mean
        residual[:, h:, :] = 0
        residual[:, :, w:] = 0
        scores = self.score_net(partition(residual, s).flatten(-2)).squeeze(-1).softmax(-1)
        self.last_ratios = self.top_k_generator(self.adapt_pool(pooled).flatten(1)).squeeze(-1)
        selections = self.select_windows(scores, self.last_ratios)

        global_tokens = pooled.flatten(2).transpose(1, 2)
        for block in self.global_blocks:
            global_tokens = block(global_tokens)
        global_map = self.score_norm(global_tokens).transpose(1, 2).reshape(b, c, hp // s, wp // s)
        global_map = F.interpolate(global_map, size=(hp, wp), mode="nearest").permute(0, 2, 3, 1)
        score_map = (
            scores.reshape(b, hp // s, wp // s).repeat_interleave(s, 1).repeat_interleave(s, 2)
        )
        score_map = score_map[:, :h, :w].reshape(b, h * w, 1)
        # Global context supplies inexpensive information outside selected windows.
        x = x + global_map[:, :h, :w].reshape(b, h * w, c) * (1 - score_map)
        for block in self.local_blocks:
            x = block(x, hw_shape, selections)
        # Keep the forward values unchanged while training the window scorer.
        x = x * (1 + score_map - score_map.detach())
        if self.downsample is None:
            return x, hw_shape, x, hw_shape
        down, down_hw = self.downsample(x, hw_shape)
        return down, down_hw, x, hw_shape


@MODELS.register_module()
class ElasticFormer(SwinTransformer):
    """Four-stage ElasticFormer with the standard MMDetection feature interface."""

    def __init__(
        self,
        embed_dims=96,
        depths=(2, 2, 6, 2),
        num_heads=(3, 6, 12, 24),
        window_size=7,
        mlp_ratio=4,
        selector_pool_sizes=(32, 16, 8, 4),
        selector_hidden_dims=(256, 128, 64, 32),
        select_gamma=0.1,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
        with_cp=False,
        **kwargs,
    ):
        super().__init__(
            embed_dims=embed_dims,
            depths=depths,
            num_heads=num_heads,
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            with_cp=with_cp,
            **kwargs,
        )
        if len(selector_pool_sizes) != len(depths) or len(selector_hidden_dims) != len(depths):
            raise ValueError("Specify one selector pooling size and hidden dimension per stage")
        drops = torch.linspace(0, drop_path_rate, sum(depths)).tolist()
        old_stages = self.stages
        stages, offset = [], 0
        for i, depth in enumerate(depths):
            stages.append(
                ElasticStage(
                    embed_dims * 2**i,
                    num_heads[i],
                    old_stages[i].downsample,
                    selector_pool_sizes[i],
                    selector_hidden_dims[i],
                    drops[offset : offset + depth],
                    window_size,
                    mlp_ratio,
                    qkv_bias,
                    qk_scale,
                    drop_rate,
                    attn_drop_rate,
                    with_cp,
                )
            )
            offset += depth
        self.stages = nn.ModuleList(stages)
        self.select_gamma = select_gamma

    def set_training_phase(self, phase):
        if phase not in (1, 2, 3):
            raise ValueError("Training phase must be 1, 2, or 3")
        for stage in self.stages:
            stage.use_dynamic_topk = phase == 3
            stage.train_selector = phase != 1
            stage.top_k_generator.requires_grad_(phase != 1)

    def selection_loss(self, targets):
        return torch.stack(
            [
                ((stage.last_ratios - targets) ** 2 + self.select_gamma * stage.last_ratios).mean()
                for stage in self.stages
            ]
        ).mean()
