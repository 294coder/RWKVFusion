"""
Positional embedding collections including:
1. 2D sine-cosine position embedding
2. Rotary Positional Encoding (RoPE)
3. Vision Transformer Need Registers pos embedding
4. Learned Axis-positonal embedding

Author: Zihan Cao
Date: 2025-01-24
Email: iamzihan666@gmail.com
License: GPL v3

---------------------------------------------------------

Copyright (c) ZihanCao, University of Electronic Science and Technology of China (UESTC), Mathematical School
"""


# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# Position embedding utils
# --------------------------------------------------------

import math
from functools import reduce
from math import ceil
from operator import mul
from typing import Any, Literal, Optional, Tuple

import numpy as np
import torch
from einops import pack, rearrange, unpack
from torch import Size, Tensor, nn, tensor
from torch.nn import Module, ModuleList


# --------------------------------------------------------
# 2D sine-cosine position embedding
# References:
# Transformer: https://github.com/tensorflow/models/blob/master/official/nlp/transformer/model_utils.py
# MoCo v3: https://github.com/facebookresearch/moco-v3
# --------------------------------------------------------
def get_2d_sincos_pos_embed(embed_dim, grid_size: tuple | int, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    if isinstance(grid_size, int):
        h, w = grid_size, grid_size
    elif isinstance(grid_size, tuple):
        h, w = grid_size
    else:
        raise ValueError("grid_size should be int or tuple")

    grid_h = np.arange(h, dtype=np.float32)
    grid_w = np.arange(w, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, h, w])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


# --------------------------------------------------------
# Interpolate position embeddings for high-resolution
# References:
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------
def interpolate_pos_embed(model, checkpoint_model):
    if "pos_embed" in checkpoint_model:
        pos_embed_checkpoint = checkpoint_model["pos_embed"]
        embedding_size = pos_embed_checkpoint.shape[-1]
        num_patches = model.patch_embed.num_patches
        num_extra_tokens = model.pos_embed.shape[-2] - num_patches
        # height (== width) for the checkpoint position embedding
        orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
        # height (== width) for the new position embedding
        new_size = int(num_patches**0.5)
        # class_token and dist_token are kept unchanged
        if orig_size != new_size:
            print(
                "Position interpolate from %dx%d to %dx%d"
                % (orig_size, orig_size, new_size, new_size)
            )
            extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
            # only the position tokens are interpolated
            pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
            pos_tokens = pos_tokens.reshape(
                -1, orig_size, orig_size, embedding_size
            ).permute(0, 3, 1, 2)
            pos_tokens = torch.nn.functional.interpolate(
                pos_tokens,
                size=(new_size, new_size),
                mode="bicubic",
                align_corners=False,
            )
            pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
            new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
            checkpoint_model["pos_embed"] = new_pos_embed


# ============================ Flux pos embedding ============================#


def rope(pos: Tensor, dim: int, theta: int) -> Tensor:
    assert dim % 2 == 0
    scale = torch.arange(0, dim, 2, dtype=torch.float64, device=pos.device) / dim
    omega = 1.0 / (theta**scale)
    out = torch.einsum("...n,d->...nd", pos, omega)
    out = torch.stack(
        [torch.cos(out), -torch.sin(out), torch.sin(out), torch.cos(out)], dim=-1
    )
    out = rearrange(out, "b n d (i j) -> b n d i j", i=2, j=2)
    return out.float()


def apply_rope(xq: Tensor, xk: Tensor, freqs_cis: Tensor) -> tuple[Tensor, Tensor]:
    xq_ = xq.float().reshape(*xq.shape[:-1], -1, 1, 2)
    xk_ = xk.float().reshape(*xk.shape[:-1], -1, 1, 2)
    xq_out = freqs_cis[..., 0] * xq_[..., 0] + freqs_cis[..., 1] * xq_[..., 1]
    xk_out = freqs_cis[..., 0] * xk_[..., 0] + freqs_cis[..., 1] * xk_[..., 1]
    return xq_out.reshape(*xq.shape).type_as(xq), xk_out.reshape(*xk.shape).type_as(xk)


class EmbedND(nn.Module):
    def __init__(self, dim: int, theta: int, axes_dim: list[int]):
        super().__init__()
        self.dim = dim
        self.theta = theta
        self.axes_dim = axes_dim

    def forward(self, ids: Tensor) -> Tensor:
        n_axes = ids.shape[-1]
        emb = torch.cat(
            [rope(ids[..., i], self.axes_dim[i], self.theta) for i in range(n_axes)],
            dim=-3,
        )

        return emb.unsqueeze(1)


# ================== Vision Transformer Need Registers pos embedding ================#


# Pos embedding
def pos_emb_sincos_2d_register(
    h, w, dim, temperature: int = 10000, dtype=torch.float32
):
    """Pos embedding for 2D image"""
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    assert (dim % 4) == 0, "dimension must be divisible by 4"

    # 1D pos embedding
    omega = torch.arange(dim // 4, dtype=dtype)
    omega = 1.0 / (temperature**omega)

    # 2D pos embedding
    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :]

    # concat sin and cos
    pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim=1).unsqueeze(0)
    return pe.type(dtype)


def interpolate_pos_embed_2d(pos_embd_ckpt: torch.Tensor, new_pos_len: int):
    embd_len_ckpt = pos_embd_ckpt.shape[1]
    ckpt_pos_hw = int(embd_len_ckpt**0.5)
    new_pos_hw = int(new_pos_len**0.5)
    ndim = pos_embd_ckpt.size(-1)

    if ckpt_pos_hw != new_pos_hw:
        pos_embd_2d = pos_embd_ckpt.reshape(-1, ckpt_pos_hw, ckpt_pos_hw, ndim).permute(
            0, 3, 1, 2
        )
        new_pos_embed = nn.functional.interpolate(
            pos_embd_2d,
            size=(new_pos_hw, new_pos_hw),
            mode="bicubic",
            align_corners=False,
        )
        return new_pos_embed.permute(0, 2, 3, 1).flatten(1, 2)
    else:
        return pos_embd_ckpt


# * Sine-cosine Positional Embedding ===========================================
## from SAM v2 repo


class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention Is All You Need paper, generalized to work on images.
    """

    def __init__(
        self,
        num_pos_feats,
        temperature: int = 10000,
        normalize: bool = True,
        scale: Optional[float] = None,
    ):
        super().__init__()
        assert num_pos_feats % 2 == 0, "Expecting even model width"
        self.num_pos_feats = num_pos_feats // 2
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

        self.cache = {}

    def _encode_xy(self, x, y):
        # The positions are expected to be normalized
        assert len(x) == len(y) and x.ndim == y.ndim == 1
        x_embed = x * self.scale
        y_embed = y * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, None] / dim_t
        pos_y = y_embed[:, None] / dim_t
        pos_x = torch.stack(
            (pos_x[:, 0::2].sin(), pos_x[:, 1::2].cos()), dim=2
        ).flatten(1)
        pos_y = torch.stack(
            (pos_y[:, 0::2].sin(), pos_y[:, 1::2].cos()), dim=2
        ).flatten(1)
        return pos_x, pos_y

    @torch.no_grad()
    def encode_boxes(self, x, y, w, h):
        pos_x, pos_y = self._encode_xy(x, y)
        pos = torch.cat((pos_y, pos_x, h[:, None], w[:, None]), dim=1)
        return pos

    encode = encode_boxes  # Backwards compatibility

    @torch.no_grad()
    def encode_points(self, x, y, labels):
        (bx, nx), (by, ny), (bl, nl) = x.shape, y.shape, labels.shape
        assert bx == by and nx == ny and bx == bl and nx == nl
        pos_x, pos_y = self._encode_xy(x.flatten(), y.flatten())
        pos_x, pos_y = pos_x.reshape(bx, nx, -1), pos_y.reshape(by, ny, -1)
        pos = torch.cat((pos_y, pos_x, labels[:, :, None]), dim=2)
        return pos

    @torch.no_grad()
    def forward(self, x: torch.Tensor):
        cache_key = (x.shape[-2], x.shape[-1])
        if cache_key in self.cache:
            return self.cache[cache_key][None].repeat(x.shape[0], 1, 1, 1)
        y_embed = (
            torch.arange(1, x.shape[-2] + 1, dtype=torch.float32, device=x.device)
            .view(1, -1, 1)
            .repeat(x.shape[0], 1, x.shape[-1])
        )
        x_embed = (
            torch.arange(1, x.shape[-1] + 1, dtype=torch.float32, device=x.device)
            .view(1, 1, -1)
            .repeat(x.shape[0], x.shape[-2], 1)
        )

        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        self.cache[cache_key] = pos[0]
        return pos


class PositionEmbeddingRandom(nn.Module):
    """
    Positional encoding using random spatial frequencies.
    """

    def __init__(self, num_pos_feats: int = 64, scale: Optional[float] = None) -> None:
        super().__init__()
        if scale is None or scale <= 0.0:
            scale = 1.0
        self.register_buffer(
            "positional_encoding_gaussian_matrix",
            scale * torch.randn((2, num_pos_feats)),
        )

    def _pe_encoding(self, coords: torch.Tensor) -> torch.Tensor:
        """Positionally encode points that are normalized to [0,1]."""
        # assuming coords are in [0, 1]^2 square and have d_1 x ... x d_n x 2 shape
        coords = 2 * coords - 1
        # [h, w, 2] @ [2, C] -> [h, w, C]
        coords = coords @ self.positional_encoding_gaussian_matrix
        coords = 2 * np.pi * coords
        # outputs d_1 x ... x d_n x C shape
        return torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)

    def forward(self, size: Tuple[int, int]) -> torch.Tensor:
        """Generate positional encoding for a grid of the specified size."""
        h, w = size
        device: Any = self.positional_encoding_gaussian_matrix.device
        grid = torch.ones((h, w), device=device, dtype=torch.float32)
        y_embed = grid.cumsum(dim=0) - 0.5
        x_embed = grid.cumsum(dim=1) - 0.5
        y_embed = y_embed / h
        x_embed = x_embed / w

        pe = self._pe_encoding(torch.stack([x_embed, y_embed], dim=-1))
        return pe.permute(2, 0, 1)  # C x H x W

    def forward_with_coords(
        self, coords_input: torch.Tensor, image_size: Tuple[int, int]
    ) -> torch.Tensor:
        """Positionally encode points that are not normalized to [0,1]."""
        coords = coords_input.clone()
        coords[:, :, 0] = coords[:, :, 0] / image_size[1]
        coords[:, :, 1] = coords[:, :, 1] / image_size[0]
        return self._pe_encoding(coords.to(torch.float))  # B x N x C


# * Rotary Positional Encoding ===================================================
# Rotary Positional Encoding, adapted from:
# 1. https://github.com/meta-llama/codellama/blob/main/llama/model.py
# 2. https://github.com/naver-ai/rope-vit
# 3. https://github.com/lucidrains/rotary-embedding-torch


def init_t_xy(end_x: int, end_y: int):
    t = torch.arange(end_x * end_y, dtype=torch.float32)
    t_x = (t % end_x).float()
    t_y = torch.div(t, end_x, rounding_mode="floor").float()
    return t_x, t_y


def compute_axial_cis(dim: int, end_x: int, end_y: int, theta: float = 10000.0):
    # 1. / temp^(range(0, dim, 4))[: (dim // 4)]
    # and norm it by dim
    freqs_x = 1.0 / (theta ** (torch.arange(0, dim, 4)[: (dim // 4)].float() / dim))
    freqs_y = 1.0 / (theta ** (torch.arange(0, dim, 4)[: (dim // 4)].float() / dim))

    # end_x: w, end_y: h
    # t_x: w*h % w
    # t_y: w*h // w
    t_x, t_y = init_t_xy(end_x, end_y)  # (l, ), (l, )
    freqs_x = torch.outer(t_x, freqs_x)  # (l, dim // 4)
    freqs_y = torch.outer(t_y, freqs_y)
    freqs_cis_x = torch.polar(
        torch.ones_like(freqs_x), freqs_x
    )  # polar length is 1, to complex tensor
    freqs_cis_y = torch.polar(torch.ones_like(freqs_y), freqs_y)
    return torch.cat([freqs_cis_x, freqs_cis_y], dim=-1)  # [l, dim // 2]


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim  # 4
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[-2], x.shape[-1])  # [dim // 2, 2]
    # [b, nh, l, c//2]
    # shape: [1, 1, l, c//2]
    shape = [d if i >= ndim - 2 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_enc(
    xq: torch.Tensor,  # [b, nh, l, c]
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,  # [l, dim // 2]
    repeat_freqs_k: bool = False,
):
    # [b, nh, l, c//2], split the channel dim
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    # [b, nh, l, c//2]
    xk_ = (
        torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
        if xk.shape[-2] != 0
        else None
    )
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(
        3
    )  # [b, nh, l, c//2] * [1, 1, l, c//2], broadcast the first dim
    if xk_ is None:
        # no keys to rotate, due to dropout
        return xq_out.type_as(xq).to(xq.device), xk

    # repeat freqs along seq_len dim to match k seq_len
    if repeat_freqs_k:
        r = xk_.shape[-2] // xq_.shape[-2]
        if freqs_cis.is_cuda:
            freqs_cis = freqs_cis.repeat(*([1] * (freqs_cis.ndim - 2)), r, 1)
        else:
            # torch.repeat on complex numbers may not be supported on non-CUDA devices
            # (freqs_cis has 4 dims and we repeat on dim 2) so we use expand + flatten
            freqs_cis = freqs_cis.unsqueeze(2).expand(-1, -1, r, -1, -1).flatten(2, 3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq).to(xq.device), xk_out.type_as(xk).to(xk.device)


# * Rotary Sine-cosine Positional Embedding ======================================
# not complex-128 freqs_cis type rope

# TODO cao: add mutli-modal rope
# links:
# https://github.com/huggingface/transformers/blob/3998fa8aab73e78de9dd9717407c4108728e8f5b/src/transformers/models/qwen2_vl/modeling_qwen2_vl.py#L102
# https://spaces.ac.cn/archives/10352


@torch.autocast(device_type="cuda", enabled=False)
def multi_modal_precompute_rope_sin_cos():
    """
    ver1. multi-modal rope (RoPE-Tie version 1) proposed by Qwen2VL (adopted ideas from Jianlin Su)

    """

    # 1. precompute the multi-modal (image, text, or video) grids

    # 2. compute the rope freqs (cos, sin) on each attention head

    pass


@torch.autocast(device_type="cuda", enabled=False)
def precompute_rope_sin_cos(
    dim: int,
    max_seq_len: list[int],
    beta_fast: float = 32.0,
    beta_slow: float = 1.0,
    rope_base: int = 10000,
    scaling_factor: float = 40.0,
    mscale: float = 1.0,
    mscale_all_dim: float = 0.0,
    *,
    original_seq_len: list[int] | None = None,
    device: torch.device = "cuda",
):
    def find_correction_dim(num_rotations, dim, base, max_seq_len):
        return (
            dim
            * math.log(max_seq_len / (num_rotations * 2 * math.pi))
            / (2 * math.log(base))
        )

    def find_correction_range(low_rot, high_rot, dim, base, max_seq_len):
        low = math.floor(find_correction_dim(low_rot, dim, base, max_seq_len))
        high = math.ceil(find_correction_dim(high_rot, dim, base, max_seq_len))
        return max(low, 0), min(high, dim - 1)

    def linear_ramp_factor(min, max, dim):
        if min == max:
            max += 0.001
        linear_func = (torch.arange(dim, dtype=torch.float32) - min) / (max - min)
        ramp_func = torch.clamp(linear_func, 0, 1)
        return ramp_func

    def yarn_get_mscale(scale=1, mscale=1):
        if scale <= 1:
            return 1.0
        return 0.1 * mscale * math.log(scale) + 1.0

    # compact the n-dim freqs
    def broadcast_to_nd(idx_in_nd: int, n_dim: int, freqs: torch.Tensor):
        # freqs: [l, d // 2]
        pattern_before = ""
        for i in range(n_dim):
            if i == idx_in_nd:
                pattern_before += " l"
            else:
                pattern_before += " 1"
        return rearrange(freqs, f"l d -> {pattern_before} d")

    # get inverse freqs
    half_dim = dim // 2
    nd = len(max_seq_len)  # number of dims
    # TODO cao: change the `half_dim` according to the dim of `max_len`, not by uniform
    inv_freqs = 1.0 / (
        rope_base
        ** (
            torch.arange(0, half_dim, nd, dtype=torch.float32, device=device) / half_dim
        )
    )
    inv_freqs_inter = 1.0 / (
        scaling_factor
        * rope_base
        ** (
            torch.arange(0, half_dim, nd, dtype=torch.float32, device=device) / half_dim
        )
    )
    inv_freqs = [inv_freqs.to(device)] * nd
    inv_freqs_inter = [inv_freqs_inter.to(device)] * nd

    # get frequences for each dimensional length
    inv_freqs_lst = []
    for di in range(nd):
        inv_freq_i = inv_freqs[di]
        inv_freq_inter_i = inv_freqs_inter[di]
        max_seq_len_i = max_seq_len[di]
        original_seq_len_i = (
            original_seq_len[di] if original_seq_len is not None else None
        )
        _get_exptrapolation_freqs = (
            max_seq_len_i > original_seq_len_i
            if original_seq_len_i is not None
            else False
        )

        # get extrapolation freqs
        if _get_exptrapolation_freqs:
            low, high = find_correction_range(
                beta_fast, beta_slow, dim, rope_base, original_seq_len_i
            )
            smooth = 1 - linear_ramp_factor(low, high, inv_freq_i.shape[-1]).to(
                device, dtype=torch.float32
            )
            inv_freq_i = inv_freq_inter_i * (1 - smooth) + inv_freq_i * smooth

        # get freqs for each dim
        t = torch.arange(max_seq_len_i, device=device, dtype=torch.float32)
        inv_freq_i = torch.outer(t, inv_freq_i)  # [l, d // 2]
        inv_freqs_lst.append(inv_freq_i)

    # broadcast to n-dim
    for di in range(nd):
        inv_freqs_lst[di] = broadcast_to_nd(di, nd, inv_freqs_lst[di]).expand(
            *max_seq_len, -1
        )  # [h, 1, d//2], [1, w, d//2] for 2d

    # concat all freqs
    freqs_all = torch.cat(inv_freqs_lst, dim=-1)  # [h, w, d] for 2d

    # stack cos, sin freqs
    _mscale = float(
        yarn_get_mscale(scaling_factor, mscale)
        / yarn_get_mscale(scaling_factor, mscale_all_dim)
    )
    freqs_cis = torch.stack(
        [torch.cos(freqs_all) * _mscale, torch.sin(freqs_all) * _mscale], dim=0
    )  # [2, h, w, d] for 2d

    return freqs_cis


@torch.autocast(device_type="cuda", enabled=False)
def interpolate_rope_sin_cos(
    freqs_cis: torch.Tensor,
    seq_len: list[int],
    interpolate_method: Literal["downsample", "star", "slice"] = "downsample",
):
    max_seq_len = tuple(freqs_cis.shape[1:-1])  # [h, w] for 2d
    nd = len(max_seq_len)
    assert len(max_seq_len) == len(
        seq_len
    ), "max_seq_len and original_seq_len must have the same length"

    # interpolate freqs
    _get_intrapolation_freqs = any(
        [
            max_seq_len_i != seq_len_i
            for max_seq_len_i, seq_len_i in zip(max_seq_len, seq_len)
        ]
        if seq_len is not None
        else False
    )
    if _get_intrapolation_freqs:
        if interpolate_method == "downsample":
            _intp_method = {3: "bicubic", 2: "bilinear", 1: "linear"}
            assert (
                nd in _intp_method
            ), f"Unknown interpolate_method: {interpolate_method}"
            freqs_cis = rearrange(freqs_cis, "b ... d -> b d ...")
            freqs_cis = torch.nn.functional.interpolate(
                freqs_cis, size=seq_len, mode=_intp_method[nd]
            )
            freqs_cis = rearrange(freqs_cis, "b d ... -> b ... d")

        # TODO: need to check
        elif interpolate_method == "star":
            indices = []
            for i in range(nd):
                max_seq_len_i = max_seq_len[i]
                seq_len_i = seq_len[i]
                reshape_dim = [1 if i != j else seq_len_i for j in range(nd)]
                index = torch.arange(seq_len_i) * (max_seq_len_i / seq_len_i).reshape(
                    *reshape_dim
                ).expand(seq_len)
                indices.append(index)
            indices = (
                torch.stack(indices, dim=-1).round().int().reshape(-1, 2)
            )  # [h * w, 2] for 2d
            freq_cis_slice = slice(None, *[indices[:, i] for i in range(nd)], None)
            freqs_cis = freqs_cis[freq_cis_slice]
            freqs_cis = freqs_cis.reshape(2, *seq_len, -1)
        elif interpolate_method == "slice":
            freqs_cis_slice = slice(None, *seq_len, None)
            freqs_cis = freqs_cis[freqs_cis_slice]
        else:
            raise ValueError(f"Unknown interpolate_method: {interpolate_method}")

        return freqs_cis


def apply_rope_sin_cos(
    q: torch.Tensor,  # [bs, h, l, d]
    k: torch.Tensor,
    freqs_cis: torch.Tensor,  # [2, *dims, d]
):
    qk = torch.stack((q, k), dim=0)  # [2, b, h, l, d]
    qk_dim_before_d = qk.shape[:-1]
    qk = qk.reshape(*qk_dim_before_d, -1, 2)  # [2, b, h, l, d//2, 2]
    freqs_cis = freqs_cis.to(qk.device).flatten(1, -2)  # [2, h*w, d//2]

    # assertions
    seq_len_qk = qk.shape[-3]
    freq_cis_len = freqs_cis.shape[-2]
    assert seq_len_qk == freq_cis_len, (
        "seq_len_qk and freq_cis_len must be the same "
        + f"but got {seq_len_qk=} and {freq_cis_len=}"
    )

    # apply freqs_cis
    cos = freqs_cis[0][None, None, None]  # [1, 1, 1, h*w, d//2]
    sin = freqs_cis[1][None, None, None]

    # [2, b, h, l, d//2, 2]
    qk = torch.stack(
        [
            cos * qk[..., 0] - sin * qk[..., 1],
            sin * qk[..., 0] + cos * qk[..., 1],
        ],
        dim=-1,
    )
    qk = qk.reshape(*qk_dim_before_d, -1)  # [2, b, h, l, d]
    q, k = qk.unbind(dim=0)  # [b, h, l, d]

    return q, k


# * Cosmos RoPE implementation ==================================================

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import List, Optional, Tuple

import torch
from einops import repeat


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


def _rotate_half_te(x: torch.Tensor) -> torch.Tensor:
    """
    change sign so the last dimension becomes [-odd, +even].
    Adopted from TransformerEngine.
    Source: https://github.com/NVIDIA/TransformerEngine/blob/main/transformer_engine/pytorch/attention.py
    """
    x = x.view(x.shape[:-1] + torch.Size((2, x.shape[-1] // 2)))
    x1, x2 = x.unbind(dim=-2)
    return torch.cat((-x2, x1), dim=-1)


def _apply_rotary_pos_emb_te(
    t: torch.Tensor,
    cos_freqs: torch.Tensor,
    sin_freqs: torch.Tensor,
) -> torch.Tensor:
    """
    Apply rotary positional embedding tensor to the input tensor.
    Adopted from TransformerEngine.
    Source: https://github.com/NVIDIA/TransformerEngine/blob/main/transformer_engine/pytorch/attention.py

    Parameters
    ----------
    t: torch.Tensor
        Input tensor of shape `[b, s, h, d]`, on which
        rotary positional embedding will be applied.
    cos_freqs: torch.Tensor
        Cosine component of rotary positional embedding tensor of shape `[s, 1, 1, d]` and dtype 'float',
    sin_freqs: torch.Tensor
        Sine component of rotary positional embedding tensor of shape `[s, 1, 1, d]` and dtype 'float',
    """
    rot_dim = cos_freqs.shape[-1]
    # ideally t_pass is empty so rotary pos embedding is applied to all tensor t
    t, t_pass = t[..., :rot_dim], t[..., rot_dim:]
    # first part is cosine component
    # second part is sine component, need to change signs with _rotate_half method
    t = (t * cos_freqs) + (_rotate_half_te(t) * sin_freqs)
    output = torch.cat((t, t_pass), dim=-1)
    return output


class RotaryPositionEmbedding(torch.nn.Module):
    """
    Rotary Position Embedding module as described in the paper:
    https://arxiv.org/abs/2104.09864

    This module implements rotary positional embeddings, which are used to
    enhance the performance of transformer models.

    Args:
        dim (int): Dimensionality of the input tensor.
        max_position_embeddings (Optional[int]): Maximum position embeddings.
        original_max_position_embeddings (Optional[int]): Original maximum position embeddings.
        rope_theta (Optional[float]): Base for the frequency calculation.
        apply_yarn (Optional[bool]): Whether to apply YaRN (Yet another Rotary).
        scale (Optional[int]): Scaling factor for the frequency calculation.
        extrapolation_factor (Optional[int]): Extrapolation factor for the frequency extension.
        attn_factor (Optional[int]): Attention factor for the frequency calculation.
        beta_fast (Optional[int]): Fast beta value for the YaRN frequency calculation.
        beta_slow (Optional[int]): Slow beta value for the YaRN frequency calculation.
        rope_dim (Optional[str]): Dimensionality of the RoPE. Choices: "1D", "2D", "3D".
        latent_shape (Optional[List[int]]): Shape of the latent tensor for video or image inputs.
        original_latent_shape (Optional[List[int]]): Original shape of the latent tensor for video or image inputs.
        pad_to_multiple_of (Optional[int]): Pad the position embedding to a multiple of this value.
    """

    def __init__(
        self,
        dim: int,
        max_position_embeddings: Optional[int] = None,
        original_max_position_embeddings: Optional[int] = None,
        rope_theta: Optional[float] = 10000.0,
        apply_yarn: Optional[bool] = False,
        scale: Optional[int] = None,
        extrapolation_factor: Optional[int] = 1,
        attn_factor: Optional[int] = 1,
        beta_fast: Optional[int] = 32,
        beta_slow: Optional[int] = 1,
        rope_dim: Optional[str] = "1D",
        latent_shape: Optional[List[int]] = None,
        original_latent_shape: Optional[List[int]] = None,
        pad_to_multiple_of: Optional[int] = None,
    ):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.original_max_position_embeddings = original_max_position_embeddings
        self.rope_theta = rope_theta
        self.apply_yarn = apply_yarn
        self.scale = scale
        self.extrapolation_factor = extrapolation_factor
        self.attn_factor = attn_factor
        self.beta_fast = beta_fast
        self.beta_slow = beta_slow
        self.mscale = 1.0
        self.rope_dim = rope_dim
        self.latent_shape = latent_shape
        self.original_latent_shape = original_latent_shape
        self.pad_to_multiple_of = pad_to_multiple_of
        self.get_inv_freq(torch.cuda.current_device())

    def get_mscale(self, scale: float = 1.0) -> float:
        """Get the magnitude scaling factor for YaRN."""
        if scale <= 1:
            return 1.0
        return 0.1 * math.log(scale) + 1.0

    def forward(self, seq_len: Optional[int] = None) -> torch.Tensor:
        """
        Forward pass for the rotary position embedding.

        Args:
            seq_len (Optional[int]): Length of the sequence.

        Returns:
            torch.Tensor: The computed frequencies for positional embedding.
        """

        if self.apply_yarn and seq_len > self.max_seq_len_cached:
            self.max_seq_len_cached = seq_len
        self.freqs = self.compute_freqs()

        return self.freqs

    def compute_freqs(
        self,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute the spatial frequencies for the latent tensor."""
        self.seq = torch.arange(self.max_seq_len_cached, dtype=torch.float).cuda()
        if self.rope_dim == "1D":
            emb = torch.einsum("i,j->ij", self.seq, self.inv_freq)

        elif self.rope_dim == "2D":
            H, W = self.latent_shape
            half_emb_h = torch.outer(self.seq[:H], self.spatial_inv_freq)
            half_emb_w = torch.outer(self.seq[:W], self.spatial_inv_freq)
            emb = torch.cat(
                [
                    repeat(half_emb_h, "h d -> h w d", w=W),
                    repeat(half_emb_w, "w d -> h w d", h=H),
                ]
                * 2,
                dim=-1,
            )
            emb = rearrange(emb, "h w d -> (h w) 1 1 d").float()

        elif self.rope_dim == "3D":
            T, H, W = self.latent_shape
            half_emb_t = torch.outer(self.seq[:T], self.temporal_inv_freq)
            half_emb_h = torch.outer(self.seq[:H], self.spatial_inv_freq)
            half_emb_w = torch.outer(self.seq[:W], self.spatial_inv_freq)
            emb = torch.cat(
                [
                    repeat(half_emb_t, "t d -> t h w d", h=H, w=W),  # d // 3
                    repeat(half_emb_h, "h d -> t h w d", t=T, w=W),  # d - (d // 3 * 2)
                    repeat(half_emb_w, "w d -> t h w d", t=T, h=H),
                ]
                * 2,
                dim=-1,
            )
            emb = rearrange(emb, "t h w d -> (t h w) 1 1 d").float()
        else:
            raise ValueError(f"Invalid RoPE dimensionality: {self.rope_dim}")
        return emb

    def get_scale_factors(
        self, inv_freq: torch.Tensor, original_seq_len: int
    ) -> torch.Tensor:
        """Get the scale factors for YaRN."""
        # Calculate the high and low frequency cutoffs for YaRN. Note: `beta_fast` and `beta_slow` are called
        # `high_freq_factor` and `low_freq_factor` in the Llama 3.1 RoPE scaling code.
        high_freq_cutoff = 2 * math.pi * self.beta_fast / original_seq_len
        low_freq_cutoff = 2 * math.pi * self.beta_slow / original_seq_len
        # Obtain a smooth mask that has a value of 0 for low frequencies and 1 for high frequencies, with linear
        # interpolation in between.
        smooth_mask = torch.clamp(
            (inv_freq - low_freq_cutoff) / (high_freq_cutoff - low_freq_cutoff),
            min=0,
            max=1,
        )
        # For low frequencies, we scale the frequency by 1/self.scale. For high frequencies, we keep the frequency.
        scale_factors = (1 - smooth_mask) / self.scale + smooth_mask
        return scale_factors

    def get_inv_freq(self, device: torch.device) -> None:
        """Get the inverse frequency."""
        if self.rope_dim == "1D":
            assert (
                self.max_position_embeddings is not None
            ), "Max position embeddings required."
            inv_freq = 1.0 / (
                self.rope_theta
                ** (
                    torch.arange(0, self.dim, 2, dtype=torch.float32, device=device)
                    / self.dim
                )
            )
            if self.apply_yarn:
                assert (
                    self.original_max_position_embeddings is not None
                ), "Original max position embeddings required."
                assert self.beta_slow is not None, "Beta slow value required."
                assert self.beta_fast is not None, "Beta fast value required."

                scale_factors = self.get_scale_factors(
                    inv_freq, self.original_max_position_embeddings
                )
                # Apply the scaling factors to inv_freq.
                inv_freq = inv_freq * scale_factors
                # Set the magnitude scaling factor.
                self.mscale = float(self.get_mscale(self.scale) * self.attn_factor)
            self.max_seq_len_cached = self.max_position_embeddings
            self.inv_freq = inv_freq

        elif self.rope_dim == "2D":
            assert self.latent_shape is not None, "Latent shape required."
            dim_h = self.dim // 2
            spatial_inv_freq = 1.0 / (
                self.rope_theta
                ** torch.arange(0, dim_h, 2, dtype=torch.float32, device=device)
                / dim_h
            )
            if self.apply_yarn:
                assert (
                    self.original_latent_shape is not None
                ), "Original latent shape required."
                assert self.beta_slow is not None, "Beta slow value required."
                assert self.beta_fast is not None, "Beta fast value required."

                scale_factors = self.get_scale_factors(
                    spatial_inv_freq, self.original_latent_shape[0]
                )
                spatial_inv_freq = spatial_inv_freq * scale_factors
                self.mscale = float(self.get_mscale(self.scale) * self.attn_factor)
            self.spatial_inv_freq = spatial_inv_freq
            self.max_seq_len_cached = max(self.latent_shape)

        elif self.rope_dim == "3D":
            assert self.latent_shape is not None, "Latent shape required."
            dim_h = self.dim // 6 * 2
            dim_t = self.dim - 2 * dim_h
            self.dim_spatial_range = (
                torch.arange(0, dim_h, 2)[: (dim_h // 2)].float().to(device) / dim_h
            )
            spatial_inv_freq = 1.0 / (self.rope_theta**self.dim_spatial_range)
            self.dim_temporal_range = (
                torch.arange(0, dim_t, 2)[: (dim_t // 2)].float().to(device) / dim_t
            )
            temporal_inv_freq = 1.0 / (self.rope_theta**self.dim_temporal_range)
            if self.apply_yarn:
                assert (
                    self.original_latent_shape is not None
                ), "Original latent shape required."
                assert self.beta_slow is not None, "Beta slow value required."
                assert self.beta_fast is not None, "Beta fast value required."
                scale_factors_spatial = self.get_scale_factors(
                    spatial_inv_freq, self.original_latent_shape[1]
                )
                spatial_inv_freq = spatial_inv_freq * scale_factors_spatial
                scale_factors_temporal = self.get_scale_factors(
                    temporal_inv_freq, self.original_latent_shape[0]
                )
                temporal_inv_freq = temporal_inv_freq * scale_factors_temporal
                self.mscale = float(self.get_mscale(self.scale) * self.attn_factor)
            self.spatial_inv_freq = spatial_inv_freq
            self.temporal_inv_freq = temporal_inv_freq
            self.max_seq_len_cached = max(self.latent_shape)
        else:
            raise ValueError(f"Invalid RoPE dimensionality: {self.rope_dim}")

        self.freqs = self.compute_freqs()


class RotaryPositionEmbeddingPytorchV2(RotaryPositionEmbedding):
    """
    Rotary Position Embedding that works in the same way as the TransformerEngine RoPE
    (https://github.com/NVIDIA/TransformerEngine/blob/main/transformer_engine/pytorch/attention.py)

    """

    def __init__(
        self,
        seq_len: int,
        training_type: str = None,
        **kwargs,
    ):
        super().__init__(
            **kwargs,
        )
        emb = self.create_rope_freqs(seq_len=seq_len, training_type=training_type)
        emb = emb.transpose(0, 1).contiguous()  # [seq, 1, 1, dim] -> [1, seq, 1, dim]
        assert emb.shape[0] == 1 and emb.shape[2] == 1, f"emb shape: {emb.shape}"
        # cos/sin first then dtype conversion for better precision
        self.register_buffer("cos_cached", torch.cos(emb), persistent=False)
        self.register_buffer("sin_cached", torch.sin(emb), persistent=False)

    def create_rope_freqs(
        self, seq_len: int, training_type: str = None
    ) -> torch.Tensor:
        """
        Create rotary position embedding frequencies.

        Args:
            seq_len (int): Sequence length of a sample.

        Returns:
            torch.Tensor: The computed positional embeddings.
        """
        if self.rope_dim == "1D":
            freqs = super().forward(seq_len=seq_len)
            emb = torch.cat((freqs, freqs), dim=-1)
            emb = emb.reshape(emb.size(0), 1, 1, emb.size(1))

        elif self.rope_dim in ["2D", "3D"]:
            emb = super().forward(seq_len=seq_len)
            if training_type == "text_to_video":
                # since we added <bov> token at the beginning of the video for text2world, we also extend the position embedding by one token in the beginning
                bov_pe = torch.zeros((1, *emb.shape[1:]), device=emb.device)
                emb = torch.cat((bov_pe, emb), dim=0)
        else:
            raise ValueError(f"Invalid RoPE dimensionality: {self.rope_dim}")
        if (
            self.pad_to_multiple_of is not None
            and emb.shape[0] % self.pad_to_multiple_of != 0
        ):
            # Round up to the nearest multiple of pad_to_multiple_of
            pad_len = self.pad_to_multiple_of - emb.shape[0] % self.pad_to_multiple_of
            emb = torch.cat(
                (emb, torch.zeros((pad_len, *emb.shape[1:]), device=emb.device)), dim=0
            )

        return emb

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        input_pos: Optional[torch.Tensor] = None,
        seq_len: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if q.dtype != self.cos_cached.dtype:
            self.cos_cached = self.cos_cached.to(q.dtype)
            self.sin_cached = self.sin_cached.to(q.dtype)

        cos_emb = self.cos_cached
        sin_emb = self.sin_cached
        if input_pos is not None:
            cos_emb = cos_emb[:, input_pos, :, :]
            sin_emb = sin_emb[:, input_pos, :, :]
        elif seq_len is not None:
            cos_emb = cos_emb[:, :seq_len, :, :]
            sin_emb = sin_emb[:, :seq_len, :, :]
        q = _apply_rotary_pos_emb_te(q, cos_emb, sin_emb)
        k = _apply_rotary_pos_emb_te(k, cos_emb, sin_emb)
        return q, k


class RotaryPositionEmbeddingPytorchV1(RotaryPositionEmbedding):
    """
    Rotary Position Embedding that works in the same way as
    mistral_inference (https://github.com/mistralai/mistral-inference/blob/main/src/mistral_inference/rope.py)
    or llama3 (https://github.com/meta-llama/llama3/blob/main/llama/model.py)

    """

    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(
            **kwargs,
        )
        if self.rope_dim == "1D":
            emb = torch.stack((self.freqs, self.freqs), dim=-1).reshape(
                *self.freqs.shape[:-1], -1
            )
        elif self.rope_dim in ["2D", "3D"]:
            emb = rearrange(self.freqs, "s 1 1 d -> s d").float()
        self.register_buffer(
            "cos_cached", (emb.cos() * self.mscale)[None, :, None, :], persistent=False
        )
        self.register_buffer(
            "sin_cached", (emb.sin() * self.mscale)[None, :, None, :], persistent=False
        )

    def rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        """Rotate half the hidden dimensions of the input tensor."""
        x_reshaped = x.reshape(*x.shape[:-1], -1, 2)
        x1 = x_reshaped[..., 0]
        x2 = x_reshaped[..., 1]
        output = torch.stack((-x2, x1), dim=-1).reshape(*x.shape)
        return output

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        input_pos: Optional[torch.Tensor] = None,
        seq_len: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for the rotary position embedding.

        Args:
            q (torch.Tensor): Query tensor.
            k (torch.Tensor): Key tensor.
            input_pos (Optional[torch.Tensor]): Starting position for the sequence.
            seq_len (Optional[int]): Length of the sequence.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Rotated query and key tensors.
        """
        if self.apply_yarn and seq_len > self.max_seq_len_cached:
            freqs = super().forward(seq_len)
            if self.rope_dim == "1D":
                emb = torch.stack((freqs, freqs), dim=-1).reshape(*freqs.shape[:-1], -1)
            elif self.rope_dim in ["2D", "3D"]:
                emb = rearrange(freqs, "s 1 1 d -> s d").float()
            else:
                raise ValueError(f"Invalid RoPE dimensionality: {self.rope_dim}")
            self.register_buffer(
                "cos_cached",
                (emb.cos() * self.mscale)[None, :, None, :].to(q.dtype),
                persistent=False,
            )
            self.register_buffer(
                "sin_cached",
                (emb.sin() * self.mscale)[None, :, None, :].to(q.dtype),
                persistent=False,
            )

        if input_pos is not None:
            cos_cached = self.cos_cached[:, input_pos]
            sin_cached = self.sin_cached[:, input_pos]
        else:
            assert (
                self.cos_cached.shape[1] >= seq_len
            ), f"Invalid sequence length; cos_cached.shape {self.cos_cached.shape}, seq_len {seq_len}."
            cos_cached = self.cos_cached[:, :seq_len, ...]
            sin_cached = self.sin_cached[:, :seq_len, ...]
        xq = q * cos_cached + self.rotate_half(q) * sin_cached
        xk = k * cos_cached + self.rotate_half(k) * sin_cached

        return xq.type_as(q), xk.type_as(k)


# * 3D Rope Positional Embedding Visualization ==================================================


def visualize_rope_sin_cos(freqs_cis: torch.Tensor):
    assert freqs_cis.dim() == 4, "Only visualize 2D rope pe"

    # Plot 3D visualization
    import matplotlib.pyplot as plt
    import numpy as np

    # Assuming freqs_cis has shape [2, h, w, d]
    # Extract cos and sin planes
    cos_planes = freqs_cis[0].cpu().numpy()  # [h, w, d]
    sin_planes = freqs_cis[1].cpu().numpy()  # [h, w, d]

    # Create meshgrid
    h, w, d = cos_planes.shape
    x, y, z = np.meshgrid(np.arange(h), np.arange(w), np.arange(d), indexing="ij")

    # Flatten data for plotting
    x = x.flatten()
    y = y.flatten()
    z = z.flatten()
    cos_values = cos_planes.flatten()
    sin_values = sin_planes.flatten()

    # Create 3D scatter plots
    fig = plt.figure(figsize=(12, 6))

    # Plot cos values in 3D
    ax1 = fig.add_subplot(121, projection="3d")
    sc1 = ax1.scatter(x, y, z, c=cos_values, cmap="viridis")
    ax1.set_title("Cos of RoPE")
    ax1.set_xlabel("Height")
    ax1.set_ylabel("Width")
    ax1.set_zlabel("Dimension")
    fig.colorbar(sc1, ax=ax1, label="Cos Value")

    # Plot sin values in 3D
    ax2 = fig.add_subplot(122, projection="3d")
    sc2 = ax2.scatter(x, y, z, c=sin_values, cmap="viridis")
    ax2.set_title("Sin of RoPE")
    ax2.set_xlabel("Height")
    ax2.set_ylabel("Width")
    ax2.set_zlabel("Dimension")
    fig.colorbar(sc2, ax=ax2, label="Sin Value")

    plt.savefig("visualized_img/rope_3d_visualization.png")
    plt.close()


# * Learnable DiT per-layer positional embedding ==============================================
# adapted from Cosmos world model (diffusion model)

from torch.nn.init import trunc_normal_


def normalize(
    x: torch.Tensor, dim: Optional[list[int]] = None, eps: float = 0
) -> torch.Tensor:
    """
    Normalizes the input tensor along specified dimensions such that the average square norm of elements is adjusted.

    Args:
        x (torch.Tensor): The input tensor to normalize.
        dim (list, optional): The dimensions over which to normalize. If None, normalizes over all dimensions except the first.
        eps (float, optional): A small constant to ensure numerical stability during division.

    Returns:
        torch.Tensor: The normalized tensor.
    """
    if dim is None:
        dim = list(range(1, x.ndim))
    norm = torch.linalg.vector_norm(x, dim=dim, keepdim=True, dtype=torch.float32)
    norm = torch.add(eps, norm, alpha=np.sqrt(norm.numel() / x.numel()))
    return x / norm.to(x.dtype)


def repeat_at_rest_dims(x: torch.Tensor, insert_dim: int, rest_lens: list[int]):
    # when insert_dim=0, rest_lens=[128, 128]
    # x: [t, d] -> [t, h, w, d] (e.g., [32, 512] -> [32, 128, 128, 512])

    non_d_dims = len(rest_lens) + 1
    for i in range(non_d_dims):
        if i != insert_dim:
            x = x.unsqueeze(i)

    expand_shape = []
    for i in range(non_d_dims):
        if i != insert_dim:
            expand_shape.append(rest_lens[i - 1])
        else:
            expand_shape.append(-1)

    return x.expand(*expand_shape, -1)


class LearnablePosAxisEmbedding(nn.Module):
    def __init__(
        self,
        dim: int,
        seq_len: list[int],
        interpolation: Literal["crop", "downsample"] = "crop",
        eps: float = 1e-6,
    ):
        super().__init__()
        self.interpolation = interpolation
        self.ndim = len(seq_len)
        self.pos_embed_n_dims = nn.ParameterList(
            [nn.Parameter(torch.zeros(seq_len[i], dim)) for i in range(self.ndim)]
        )
        self.eps = eps

        # trunc normal init
        for i in range(self.ndim):
            trunc_normal_(self.pos_embed_n_dims[i], std=0.02)

    def forward(self, axials: tuple[int, ...]):
        # x: [b, *ndims, d]

        ndim_context = len(axials)
        assert (
            len(self.pos_embed_n_dims) == ndim_context
        ), f"len(self.pos_embed) must be equal to ndim_context, but got {len(self.pos_embed_n_dims)=} and {ndim_context=}"

        # interpolate
        if self.interpolation == "crop":
            pos_embed = torch.zeros(
                1,
            )  ## FIXME: get the shape of pos_embed
            for i in range(self.ndim):
                len_i = x.shape[i + 1]
                embed_i = self.pos_embed_n_dims[i][:len_i]  # [len_i, d]
                # repeat at the rest of context dims
                embed_i = repeat_at_rest_dims(embed_i, i, axials.pop(i))
                pos_embed = pos_embed + embed_i
        elif self.interpolation == "downsample":
            pos_embed = [
                self.pos_embed_n_dims[i][:: x.shape[i + 1]] for i in range(ndim_context)
            ]
        else:
            raise ValueError(f"Unknown interpolation method: {self.interpolation}")

        return normalize(pos_embed, dim=-1, eps=self.eps)


# helper functions


def exists(v):
    return v is not None


class AxialPositionalEmbedding(nn.Module):
    def __init__(
        self,
        dim,
        axial_shape: tuple[int, ...],
        axial_dims: tuple[int, ...] | None = None,
    ):
        super().__init__()

        self.dim = dim
        self.shape = axial_shape
        self.max_seq_len = reduce(mul, axial_shape, 1)

        self.summed = not exists(axial_dims)
        axial_dims = ((dim,) * len(axial_shape)) if self.summed else axial_dims

        assert len(self.shape) == len(
            axial_dims
        ), "number of axial dimensions must equal the number of dimensions in the shape"
        assert (
            self.summed or not self.summed and sum(axial_dims) == dim
        ), f"axial dimensions must sum up to the target dimension {dim}"

        self.weights = nn.ParameterList([])

        for ind, (shape, axial_dim) in enumerate(zip(self.shape, axial_dims)):
            ax_shape = [1] * len(self.shape)
            ax_shape[ind] = shape
            ax_shape = (1, *ax_shape, axial_dim)  # [1, h, w, d]
            ax_emb = nn.Parameter(
                torch.zeros(ax_shape)
            )  # in the original implementation, they use normal_(0, 1)
            ax_emb = nn.init.trunc_normal_(ax_emb, std=0.02)
            self.weights.append(ax_emb)

    def forward(self, x):
        batch, seq_len, _ = x.shape
        assert (
            seq_len <= self.max_seq_len
        ), f"Sequence length ({seq_len}) must be less than the maximum sequence length allowed ({self.max_seq_len})"

        embs = []

        for ax_emb in self.weights:
            axial_dim = ax_emb.shape[-1]
            expand_shape = (batch, *self.shape, axial_dim)
            emb = ax_emb.expand(expand_shape).reshape(
                batch, self.max_seq_len, axial_dim
            )
            embs.append(emb)

        pos_emb = sum(embs) if self.summed else torch.cat(embs, dim=-1)

        return pos_emb[:, :seq_len].to(x)


# wrapper for images


class AxialPositionalEmbedding2D(nn.Module):
    def __init__(
        self,
        dim,
        axial_shape: tuple[int, ...],
        axial_dims: tuple[int, ...] | None = None,
    ):
        super().__init__()
        assert len(axial_shape) == 2, "Axial shape must have 2 dimensions for images"
        self.pos_emb = AxialPositionalEmbedding(dim, axial_shape, axial_dims)

    def forward(self, img):
        img = rearrange(img, "b c h w -> b h w c")
        img, packed_shape = pack([img], "b * c")

        pos_emb = self.pos_emb(img)

        (pos_emb,) = unpack(pos_emb, packed_shape, "b * c")
        pos_emb = rearrange(pos_emb, "b h w c -> b c h w")
        return pos_emb


# * learnable factorized axisal positional embedding ==========================================================


class LearnablePosAxisEmbedding2D(nn.Module):
    def __init__(
        self,
        dim: int,
        seq_len: list[int, int],
        factorize: bool = True,
        interpolation: Literal["crop", "interpolate"] = "crop",
        eps: float = 1e-6,
    ):
        super().__init__()
        self.interpolation = interpolation
        self.factorize = factorize
        assert interpolation in [
            "crop",
            "interpolate",
        ], "Unknown interpolation method, only support `crop` or `interpolate`"
        assert len(seq_len) == 2, "seq_len must be a list of 2 integers"
        self.max_h, self.max_w = seq_len
        if factorize:
            self.pos_embed_h = nn.Parameter(torch.zeros(dim, self.max_h))
            self.pos_embed_w = nn.Parameter(torch.zeros(dim, self.max_w))
            # trunc normal init
            trunc_normal_(self.pos_embed_h, std=0.02)
            trunc_normal_(self.pos_embed_w, std=0.02)
        else:
            self.pos_embed = nn.Parameter(torch.zeros(1, dim, self.max_h, self.max_w))
            trunc_normal_(self.pos_embed, std=0.02)
        self.eps = eps

    def forward_factorize(self, hw: tuple[int, int]):
        h, w = hw

        # interpolate
        _force_to_interp = h > self.max_h or w > self.max_w
        if self.interpolation == "crop" and not _force_to_interp:
            assert (
                h <= self.max_h and w <= self.max_w
            ), "input height and width must be less than or equal to the positional embedding height and width"
            pos_embed_h = self.pos_embed_h[:, :h]
            pos_embed_w = self.pos_embed_w[:, :w]
            # repeat
            embed = repeat(pos_embed_h, "d h -> b d h w", b=1, w=w) + repeat(
                pos_embed_w, "d w -> b d h w", b=1, h=h
            )
        elif self.interpolation == "interpolate" or _force_to_interp:
            _interp_fn = lambda x: torch.nn.functional.interpolate(
                x, size=(h, w), mode="bilinear", align_corners=True
            )
            embed = _interp_fn(
                repeat(self.pos_embed_h, "d h -> b d h w", b=1, w=w)
            ) + _interp_fn(repeat(self.pos_embed_w, "d w -> b d h w", b=1, h=h))
        else:
            raise ValueError(
                f"Unknown interpolation method: {self.interpolation}, or input height and width must be less than or equal to the positional embedding height and width"
            )

        # normalize
        embed = normalize(embed, dim=1, eps=self.eps)

        return embed

    def forward_non_factorize(self, hw: tuple[int, int]):
        h, w = hw

        if self.interpolation == "crop":
            pos_embed = self.pos_embed[:, :h, :w]
        elif self.interpolation == "interpolate":
            self.pos_embed
            pos_embed = torch.nn.functional.interpolate(
                self.pos_embed, size=(h, w), mode="bilinear"
            )

        return normalize(pos_embed, dim=-1)

    def forward(self, axial_dim: tuple[int, ...], flatten: bool = True):
        if self.factorize:
            pe = self.forward_factorize(axial_dim)
        else:
            pe = self.forward_non_factorize(axial_dim)

        if flatten:
            pe = rearrange(pe, "b d h w -> b (h w) d")

        return normalize(pe, dim=-1)


# * Continous MLP learnable factorized positional embedding ==========================================================

# mlp - continuously parameterizing each axial position


def MLP(dim_in, dim_out, depth=2, expansion=2):
    curr_dim = dim_in
    dim_hidden = int(expansion * max(dim_in, dim_out))

    layers = []

    for _ in range(depth):
        layers.append(nn.Linear(curr_dim, dim_hidden))
        layers.append(nn.SiLU())

        curr_dim = dim_hidden

    layers.append(nn.Linear(curr_dim, dim_out))
    return nn.Sequential(*layers)


# main class


class ContinuousAxialPositionalEmbedding(Module):
    def __init__(
        self,
        dim,
        axials: tuple[int, ...] | None = None,
        num_axial_dims: int | None = None,
        mlp_depth: int = 2,
        mlp_expansion: int = 2.0,
        interp_type: str = "linear",
    ):
        """
        ## Usage

        >>> import torch
        >>> from axial_positional_embedding import (
        ...     ContinuousAxialPositionalEmbedding,
        ... )
        >>> pos_emb = ContinuousAxialPositionalEmbedding(
        >>>     dim = 512,
        >>>     num_axial_dims = 3
        >>> )
        >>> tokens = torch.randn(
        ...     1,
        ...     8,
        ...     16,
        ...     32,
        ...     512,
        ... )  # say a video with 8 frames, 16 x 32 image dimension
        >>> axial_pos_emb = pos_emb(
        ...     (8, 16, 32)
        ... )  # pass in the size from above
        >>> tokens = (
        ...     axial_pos_emb
        ...     + tokens
        ... )  # add positional embedding to token embeddings
        """
        super().__init__()
        if exists(axials):
            self.num_axial_dims = len(axials)
        elif exists(num_axial_dims):
            self.num_axial_dims = num_axial_dims
        else:
            raise ValueError(
                "either axials or num_axial_dims can not be None at the same time"
            )

        # mlps for each axial dimension
        self.mlps = ModuleList(
            [
                MLP(1, dim, depth=mlp_depth, expansion=mlp_expansion)
                for _ in range(self.num_axial_dims)
            ]
        )
        # dummy buffer for device and dtype
        self.register_buffer("dummy", tensor(0), persistent=False)

        # max sequence length
        self.interp_type = interp_type
        max_seq_len = axials
        if max_seq_len is not None:
            assert (
                len(max_seq_len) == self.num_axial_dims
            ), "max_seq_len must have the same length as the number of axial dimensions"
            self.register_buffer(
                "max_seq_len", torch.tensor(max_seq_len)
            )  # may affect EMA
            # self.max_seq_len = max_seq_len
        else:
            self.max_seq_len = None

    @property
    def device(self):
        return self.dummy.device

    @property
    def dtype(self):
        return next(self.mlps.parameters()).dtype

    def combine_factorized(
        self,
        axial_embeds: list[Tensor],
        axial_dims: tuple[int, ...] | None = None,
        flatten=False,
    ):
        if not exists(axial_dims):
            axial_dims = tuple(axial_embed.shape[0] for axial_embed in axial_embeds)

        assert len(axial_dims) == len(axial_embeds)

        axial_embeds = [
            axial_embed[:axial_dim]
            for axial_embed, axial_dim in zip(axial_embeds, axial_dims)
        ]

        axial_embed, *rest_axial_embeds = axial_embeds

        for rest_axial_embed in rest_axial_embeds:
            axial_embed = axial_embed[..., None, :] + rest_axial_embed

        assert axial_embed.shape[:-1] == axial_dims

        if flatten:
            axial_embed = rearrange(axial_embed, "... d -> (...) d")

        return axial_embed

    def maybe_derive_outer_dim(
        self, max_seq_len, axial_dims: Tensor | Size | tuple[int, ...]
    ):
        ndims = self.num_axial_dims
        assert len(axial_dims) in (ndims, ndims - 1)

        if len(axial_dims) == ndims:
            return axial_dims

        stride = reduce(mul, (*axial_dims,))

        outer_dim = ceil(max_seq_len / stride)
        return (outer_dim, *axial_dims)

    def forward_with_seq_len(
        self,
        seq_len: int,
        axial_dims: Tensor | Size | tuple[int, ...] = (),
        *,
        factorized: list[Tensor] | None = None,
        return_factorized=False,
    ):
        if not exists(factorized):
            axial_dims = self.maybe_derive_outer_dim(seq_len, axial_dims)
            factorized = self.forward(axial_dims, return_factorized=True)

        axial_embeds = self.combine_factorized(factorized, flatten=True)

        axial_embeds = axial_embeds[:seq_len]

        if not return_factorized:
            return axial_embeds

        return axial_embeds, factorized

    def forward_with_pos(
        self,
        pos: Tensor,
        axial_dims: Tensor | Size | tuple[int, ...] = (),
    ):
        assert pos.dtype in (torch.int, torch.long)

        max_pos = pos.amax().item() + 1
        axial_dims = self.maybe_derive_outer_dim(max_pos, axial_dims)
        indices = torch.unravel_index(pos, axial_dims)

        axial_embed = 0.0

        for mlp, axial_index in zip(self.mlps, indices):
            axial_index = rearrange(axial_index, "... -> ... 1")
            axial_embed = axial_embed + mlp(axial_index.to(self.dtype))

        return axial_embed

    def make_seq_len_for_mlp(self, axial_dim: int, dim_i: int):
        max_seq_len = self.max_seq_len[dim_i] if self.max_seq_len is not None else None

        if (
            max_seq_len is None
            or self.interp_type == "unchange"
            or axial_dim <= max_seq_len
        ):
            embed = torch.arange(axial_dim, device=self.device, dtype=self.dtype)
        elif axial_dim > max_seq_len:
            if self.interp_type == "linear":
                embed = torch.linspace(
                    0,
                    max_seq_len - 1,
                    steps=axial_dim,
                    device=self.device,
                    dtype=self.dtype,
                )
            else:
                raise ValueError(
                    f"Unknown interpolation type: {self.interp_type}, only support `linear`"
                )
        else:
            raise NotImplementedError(
                "max_seq_len = {} and interp_type = {}".format(
                    max_seq_len, self.interp_type
                )
            )

        assert (
            embed.shape[0] == axial_dim
        ), "axial dim must be equal to the length of the embedding, not got {} and {}".format(
            axial_dim, embed.shape[0]
        )

        return embed

    def forward(
        self,
        axial_dims: Tensor | Size | tuple[int, ...] | None = None,
        return_factorized=False,  # whether to return list[Tensor] of factorized axial positional embeddings
        flatten=True,  # whether to flatten axial dims
        align_batch_size=True,  # whether to repeat the batch size
    ):
        axial_embeds = []

        for i, (mlp, axial_dim) in enumerate(zip(self.mlps, axial_dims)):
            seq = self.make_seq_len_for_mlp(axial_dim, i)
            axial_embed = mlp(rearrange(seq, "n -> n 1"))

            axial_embeds.append(axial_embed)

        if return_factorized:
            assert not flatten

            # needed for Transfusion
            return axial_embeds

        axial_embed = self.combine_factorized(axial_embeds, flatten=flatten)

        if align_batch_size:
            axial_embed = axial_embed.unsqueeze(0)

        return axial_embed


if __name__ == "__main__":
    # * test 2D sine-cosine pos embedding
    # pos_embed = pos_emb_sincos_2d_register(32, 32, 128)
    # x = torch.randn(2, 1024, 128)
    # x = x + pos_embed

    # pos_embed2 = interpolate_pos_embed_2d(pos_embed, 64*64)
    # print(pos_embed2.shape)

    # head_dim = 64
    # seq_len = [64, 64]
    # original_seq_len = [32, 32]
    # rope_base = 10000
    # freq_cis_orig = precompute_rope_sin_cos(head_dim, original_seq_len, 32.0, 1.0, rope_base=rope_base,
    #                                         scaling_factor=1.0)  # [2, h, w, d]

    # freqs_cis = precompute_rope_sin_cos(head_dim, seq_len, 32.0, 1.0, rope_base=rope_base,
    #                                     scaling_factor=1.0, original_seq_len=original_seq_len)  # [2, h, w, d]
    # print(freqs_cis.shape)

    # # apply the rope
    # q = torch.randn(2, 8, 64 * 64, head_dim)
    # k = torch.randn(2, 8, 64 * 64, head_dim)
    # q, k = apply_rope_sin_cos(q, k, freqs_cis)
    # print(q.shape, k.shape)

    # # visualize the rope
    # visualize_rope_sin_cos(freqs_cis)

    # * test axis learnable positional embedding
    # lets say an image with 64x64 pixels
    # seq_len = [64, 64]
    # pos_embed = LearnablePosAxisEmbedding2D(32, seq_len)
    # x = torch.randn(2, 3, 64, 64)
    # print(pos_embed(x).shape)

    # axis_pos_pe = AxialPositionalEmbedding2D(32, (64, 64))
    # axis_pos_pe = ContinuousAxialPositionalEmbedding(3, (32, 32))
    # axis_pos_pe = LearnablePosAxisEmbedding2D(32, (64, 64), factorize=True)

    # img = torch.randn(2, 3, 64, 64)
    # pe = axis_pos_pe(img.shape[-2:], flatten=True)

    # print(pe.shape)

    # * test Cosmos RoPE embedding

    rope = RotaryPositionEmbeddingPytorchV2(
        seq_len=32 * 32,
        dim=32,
        apply_yarn=True,
        scale=2.0,
        latent_shape=(64, 64),
        original_latent_shape=(32, 32),
        rope_dim="2D",
        beta_fast=4,
        beta_slow=1,
    ).cuda()

    x = torch.randn(2, 3, 64, 64)
    print(rope.create_rope_freqs(64 * 64).shape)

    q = torch.randn(2, 64 * 64, 8, 32).cuda()
    k = torch.randn(2, 64 * 64, 8, 32).cuda()

    q, k = rope(q, k)

    print(q.shape)
