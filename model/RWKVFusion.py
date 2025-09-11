from dataclasses import dataclass
from functools import partial
from typing import Callable, Literal, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from kornia.filters import sobel as sobel_op
from timm.layers.drop import DropPath
from torch.nn.init import trunc_normal_
from typeguard import typechecked

from model.base_model import BaseModel, register_model
from model.module import PatchMergeModule
from model.module.layer_norm import LayerNorm, RMSNorm
from model.module.pos_embedding import (
    get_1d_sincos_pos_embed_from_grid,
    get_2d_sincos_pos_embed,
)
from model.module.rwkv_v5_multi_modal import (
    HEAD_SIZE,
    CrossMerge,
    CrossMergeTriton,
    CrossMergeTritonSelect,
    CrossScan,
    CrossScanTriton,
    CrossScanTritonSelect,
    cross_merge_k0,
    cross_scan_k0,
)
from model.module.rwkv_v5_multi_modal import VRWKV_ChannelMix_wkv5 as RWKV_CMix
from model.module.rwkv_v5_multi_modal import VRWKV_SpatialMix_wkv5 as RWKV_TMix
from utils import catch_any_error, easy_logger

logger = easy_logger(func_name="RWKVFusion", level="INFO")

#####################################################################################
# history
# 2024-10-24: convert all tensor shape into C-last to align with Vision-RWKV
# 2025-09-11: Reformat all codes

#####################################################################################


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, (H, W) = x.shape[0], x.shape[-2:]
    assert (
        H % window_size == 0 and W % window_size == 0
    ), "image height and width must be divisible by window size"

    # c-last sequence
    windows = rearrange(
        x, "b c (nh p1) (nw p2) -> (b nh nw) c p1 p2", p1=window_size, p2=window_size
    )
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = rearrange(
        windows,
        "(b nh nw) c p1 p2 -> b c (nh p1) (nw p2)",
        p1=window_size,
        p2=window_size,
        b=B,
        nh=H // window_size,
        nw=W // window_size,
    )
    return x


# *==============================================================
# * Layers
# *==============================================================


class Modulator(nn.Module):
    def __init__(self, dim: int, double: bool = True, K: int = 1):
        super().__init__()
        self.double = double
        self.multip = 6 if double else 3
        self.K = K

        self.modulated = nn.Linear(dim, dim * self.multip, bias=False)

        # if self.K == 1:
        #     self.modulated = nn.Linear(dim, dim * self.multip, bias=False)
        # else:
        #     self.modulated = nn.Linear(dim // K, dim * self.multip, bias=False)

    def forward(self, x):
        # if self.K == 1:
        #     return self.modulated(F.silu(x)).chunk(self.multip, dim=-1)
        # else:
        #     # [b, l, k x d]
        #     x = F.silu(x)
        #     x = x.view(x.size(0), x.size(1), self.K, -1)
        #     # [b, l, k, d] @ [d, k x m] -> [b, l, k, m]
        #     x = torch.einsum('blkd,cd->blkc', x, self.modulated.weight)
        #     x = rearrange(x, 'b l k (m d) -> b l (k d) m', m=self.multip)
        #     return x.unbind(dim=-1)

        return self.modulated(F.silu(x)).chunk(self.multip, dim=-1)


class FeedForwardLLaMA(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
        ffn_dim_multiplier: float | None = None,
        *,
        feature_ndim: int = 3,
    ):
        super().__init__()
        # custom dim factor multiplier
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        if feature_ndim == 3:
            self.w1 = nn.Linear(dim, hidden_dim, bias=False)
            self.w2 = nn.Linear(dim, hidden_dim, bias=False)
            self.w3 = nn.Linear(dim, hidden_dim, bias=False)
        elif feature_ndim == 4:
            self.w1 = nn.Conv2d(dim, hidden_dim, 1, bias=False)
            self.w2 = nn.Conv2d(dim, hidden_dim, 1, bias=False)
            self.w3 = nn.Conv2d(dim, hidden_dim, 1, bias=False)
        else:
            raise ValueError(f"Unsupported feature dimension: {feature_ndim}")

    def forward(self, x):
        # LLaMA3 type FFN
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class FeedForwardQWen2(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
        ffn_dim_multiplier: float | None = None,
        *,
        feature_ndim: int = 4,
    ):
        super().__init__()
        # custom dim factor multiplier
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        if feature_ndim == 3:
            self.w1 = nn.Linear(dim, hidden_dim, bias=False)
            self.w2 = nn.Linear(dim, hidden_dim, bias=False)
        elif feature_ndim == 4:
            self.w1 = nn.Conv2d(dim, hidden_dim, 1, bias=False)
            self.w2 = nn.Conv2d(dim, hidden_dim, 1, bias=False)
        else:
            raise ValueError(f"Unsupported feature dimension: {feature_ndim}")

    def forward(self, x):
        # QWen2VL type FFN
        return self.w2(F.gelu(self.w1(x), approximate="tanh"))


@dataclass
class ConditionInput:
    modalities: torch.Tensor | tuple[torch.Tensor, torch.Tensor] = (
        None  # cat two modalities
    )
    llm_feature: torch.Tensor = None  # llm feature
    mask_input: torch.Tensor = None  # mask input

    def __post_init__(self):
        assert self.modalities is not None, "two modalities must be involved"
        if isinstance(self.modalities, tuple):
            assert len(self.modalities) == 2, "two modalities must be involved"
            self.modalities = torch.cat(self.modalities, dim=1)

    def __repr__(self):
        return (
            f"modalities: {self.modalities.shape}, "
            + f"llm_feature: {self.llm_feature.shape if self.llm_feature is not None else None}, "
            + f"mask_input: {self.mask_input.shape if self.mask_input is not None else None}, "
        )


# ============================== Norm cfgs ===============================

norm_cls_channel_first = partial(
    LayerNorm, LayerNorm_type="WithBias", channel_first=True
)
norm_cls_channel_last = partial(
    LayerNorm, LayerNorm_type="WithBias", channel_first=False
)

# norm_cls_channel_first = partial(RMSNorm, channel_first=True)
# norm_cls_channel_last = partial(RMSNorm, channel_first=False)

# ============================== Some global cfgs ===============================
INTERP_ALIGN_CORNERS = False  # False for Pansharpening
INTERP_ANTIALIAS = False
CONV_PAD_MODE = "reflect"
T_MIX_DECAY_SPEED = "fast"
ATTN_KEY_NORM = False  # False to use l2 norm, True to use LayerNorm
FFN_KEY_NORM = False  # False not use any norm, True to use LayerNorm

# print
logger.info(
    f"{INTERP_ALIGN_CORNERS=}, {INTERP_ANTIALIAS=}",
    f"{CONV_PAD_MODE=}",
    f"{T_MIX_DECAY_SPEED=}",
    f"{FFN_KEY_NORM=}",
    f"{ATTN_KEY_NORM=}",
)


def interpolate(
    interp_type: str,
    img: torch.Tensor,
    factor: float = None,
    img_sz: tuple[int, int] = None,
    align_corners: bool = INTERP_ALIGN_CORNERS,
    antialias: bool = INTERP_ANTIALIAS,
):
    if interp_type == "nearest":
        align_corners = None
        antialias = None
    # assert not antialias, "antialias is not supported"

    assert factor is not None or img_sz is not None, "factor or img_sz must be provided"

    return F.interpolate(
        img,
        size=img_sz,
        scale_factor=factor,
        mode=interp_type,
        align_corners=align_corners,
        antialias=antialias,
    )


# *==============================================================
# * Blocks
# *==============================================================


class MultiModalityFusion(nn.Module):
    """
    MIFM for U-Net conditions pre-fusion
    """

    def __init__(
        self,
        n_embd: int,
        modal_chan: int,
        mask_chan: int = None,
        llm_chan: int = None,
        feat_drop: float = 0.0,
        llm_drop: float = 0.0,
        llm_pe_type: Literal["none", "sine"] = "none",
        img_pe_type: Literal["none", "sine"] = "none",
        scan_mode: str = "K2",
        ffn_type: Literal["llama", "qwen2"] = "qwen2",
        *,
        # multi-modal RWKV configure
        layer_id: int = 0,
        n_layer: int = 8,
        scan_id: int = 0,
        shift_type: str = "none",
        attn_bias: bool = False,
        img_ds_r_by_llm: int = 4,
        multi_modal_mlp_hidden_ratio: int = 2,
        last_enc_block: bool = False,
        add_mm_tokens: bool = False,
        checkpoint: bool = False,
    ):
        super().__init__()

        self.n_embd = n_embd
        self.modal_chan = modal_chan
        self.mask_chan = mask_chan
        self.llm_chan = llm_chan
        self.img_ds_r_by_llm = img_ds_r_by_llm
        self.add_mm_tokens = add_mm_tokens
        self.with_mask = False
        self.with_llm_feat = False
        self.txt_cat_order = scan_id % 2
        self.last_enc_block = last_enc_block

        assert (
            self.img_ds_r_by_llm >= 1
        ), "img_ds_r_by_llm should be greater than or equal to 1"

        # previous feature encoder
        self.feat_norm = norm_cls_channel_first(n_embd)
        self.feat_convs = nn.Sequential(
            nn.Conv2d(
                n_embd,
                n_embd,
                kernel_size=3,
                padding=1,
                groups=n_embd,
                bias=False,
                padding_mode=CONV_PAD_MODE,
            ),
            nn.Conv2d(n_embd, n_embd, 1, bias=False),
            nn.LeakyReLU(negative_slope=0.2),
            norm_cls_channel_first(n_embd),
        )

        # modalities encoder
        self.modal_convs = nn.Conv2d(
            modal_chan,
            n_embd,
            kernel_size=3,
            padding=1,
            bias=False,
            padding_mode=CONV_PAD_MODE,
        )
        self.modal_norm = norm_cls_channel_first(n_embd)
        self.adap_pool = nn.AdaptiveAvgPool2d(1)

        # modals bottleneck
        # botneck_ratio = 2
        self.act_conv = nn.Sequential(
            # nn.Conv2d(n_embd, n_embd // botneck_ratio, 3, 1, 1, bias=False),
            # norm_cls_channel_first(n_embd // botneck_ratio),
            # nn.Conv2d(n_embd // botneck_ratio, n_embd * 2, 1, 1, 0, bias=False),
            nn.Conv2d(n_embd, n_embd, 1, 1, 0, bias=False),
            # nn.SiLU(),
            nn.Sigmoid(),
        )
        self.feat_drop = nn.Dropout(feat_drop) if feat_drop > 0.0 else nn.Identity()
        self.fusion_proj = nn.Sequential(
            norm_cls_channel_first(n_embd),
            nn.Conv2d(n_embd, n_embd, kernel_size=1),
        )

        # mask prompt encoder
        if mask_chan is not None:
            self.with_mask = True
            mask_ratio = 2
            # progressive encoding
            self.mask_convs = nn.Sequential(
                nn.Conv2d(mask_chan, n_embd // mask_ratio, kernel_size=1, bias=False),
                nn.Conv2d(
                    n_embd // mask_ratio,
                    n_embd,
                    kernel_size=3,
                    padding=1,
                    bias=False,
                    padding_mode=CONV_PAD_MODE,
                ),
                norm_cls_channel_first(n_embd),
                # ver1:
                nn.SiLU(),
                # ver2:
                # nn.Sigmoid(),
            )

        # modals ln
        # self.modals_out_ln = norm_cls_channel_first(n_embd)

        # llm feature encoder
        if llm_chan is not None:
            self.with_llm_feat = True

            K = int(scan_mode[1]) or 1

            # use <img> and <txt> token
            if self.add_mm_tokens:
                assert (
                    self.llm_chan is not None
                ), "add_mm_tokens must be True when using llm feature"
                self.img_token = nn.Parameter(torch.zeros(2, 1, n_embd * K))
                self.txt_token = nn.Parameter(torch.zeros(2, 1, n_embd * K))

            # linear head for llm feature
            self.llm_dense = nn.Sequential(
                norm_cls_channel_last(llm_chan), nn.Linear(llm_chan, n_embd, bias=False)
            )
            self.llm_drop = nn.Dropout(llm_drop) if llm_drop > 0.0 else nn.Identity()
            assert scan_mode in [
                "K0",
                "K2",
                "K4",
                "K8",
            ], "scan_mode should be one of [K2, K4, K8]"
            self.K = K
            n_embd = n_embd
            dim_att = n_embd
            N_HEAD = n_embd // HEAD_SIZE

            # multi-modal RWKV attention
            self.multi_modal_attn = RWKV_TMix(
                dim_att,
                N_HEAD,
                n_layer,
                layer_id,
                shift_mode=shift_type,
                attn_bias=attn_bias,
                key_norm=ATTN_KEY_NORM,
                with_cp=checkpoint,
                img_txt_cat_order=self.txt_cat_order,
                scan_K=K,
                decay_speed=T_MIX_DECAY_SPEED,
            )
            self.multi_modal_img_ln = norm_cls_channel_last(n_embd)
            self.multi_modal_txt_ln = norm_cls_channel_last(n_embd)

            # img and txt out norm
            # self.llm_out_ln = norm_cls_channel_first(n_embd)

            # image and text mlps
            ffn_cls = FeedForwardLLaMA if ffn_type == "llama" else FeedForwardQWen2
            self.img_mlp = ffn_cls(
                n_embd, n_embd, multi_modal_mlp_hidden_ratio, feature_ndim=4
            )
            if not self.last_enc_block:
                self.txt_mlp = ffn_cls(
                    n_embd, n_embd, multi_modal_mlp_hidden_ratio, feature_ndim=3
                )

            # ESS
            if scan_mode == "K0":
                self.scan = cross_scan_k0
                self.merge = cross_merge_k0
            elif scan_mode == "K2":
                self.scan = lambda x: CrossScanTritonSelect.apply(x, scan_id % 2)
                self.merge = lambda x: CrossMergeTritonSelect.apply(x, scan_id % 2)
            elif scan_mode == "K4":
                self.scan = CrossScanTriton.apply
                self.merge = CrossMergeTriton.apply
            else:  # K8
                self.scan = CrossScan.apply
                self.merge = CrossMerge.apply

        # postional encoding
        self.img_pe_type = img_pe_type
        self.llm_pe_type = llm_pe_type

    def add_img_pe(self, img, h, w):
        bs = img.size(0)

        if self.img_pe_type == "sine":
            img_pe = torch.from_numpy(get_2d_sincos_pos_embed(self.n_embd, (h, w))).to(
                img
            )
            # add pe
            img = img + img_pe.unsqueeze(0).repeat(bs, 1, 1, 1)

        return img

    def add_llm_pe(self, llm_txt):
        if llm_txt is None:
            return llm_txt

        bs, d, l = llm_txt.shape

        if self.llm_pe_type == "sine" and llm_txt is not None:
            llm_ids = np.arange(l, dtype=np.float32)
            llm_pe = torch.from_numpy(
                get_1d_sincos_pos_embed_from_grid(self.n_embd, llm_ids)
            ).to(llm_txt)
            # add pe
            llm_txt = llm_txt + llm_pe.unsqueeze(0).repeat(bs, 1, 1)

        return llm_txt

    def extract_img_and_llm_feat(
        self, h: int, w: int, att_or_ffn_out: torch.Tensor, has_llm: bool
    ):
        # att_or_ffn_out: [bs, K, C, L]
        if has_llm:
            if self.add_mm_tokens:
                # <start_img> img_tokens <end_img> <start_txt> txt_tokens <end_txt>
                if self.txt_cat_order == 0:
                    slice_1 = slice(1, h * w + 1)  # img_feat
                    slice_2 = slice(h * w + 3, -1)  # llm_feat
                else:
                    slice_1 = slice(-(h * w + 1), -1)  # llm_feat
                    slice_2 = slice(1, -(h * w + 3))  # img_feat
                img_feat, llm_feat = (
                    att_or_ffn_out[:, slice_1],
                    att_or_ffn_out[:, slice_2],
                )
            else:
                if self.txt_cat_order == 0:
                    slice_1 = slice(None, h * w)
                    slice_2 = slice(h * w, None)
                else:
                    slice_1 = slice(-(h * w), None)
                    slice_2 = slice(None, -(h * w))
                img_feat, llm_feat = (
                    att_or_ffn_out[:, slice_1],
                    att_or_ffn_out[:, slice_2],
                )
        else:
            img_feat = att_or_ffn_out
            llm_feat = None

        return img_feat, llm_feat

    def forward(self, img_feat: torch.Tensor, cond_input: ConditionInput):
        modals = cond_input.modalities
        mask = cond_input.mask_input
        llm_feat = cond_input.llm_feature

        # modalities encoder
        modals = self.modal_convs(modals)
        modals = self.modal_norm(modals)

        # feature encoder
        img_feat = self.feat_drop(self.feat_convs(img_feat))

        # feature gate
        img_gate = self.feat_norm(img_feat + modals)
        feat_pool = self.adap_pool(img_gate)
        feat_mul = self.act_conv(feat_pool)  # to gate
        # m1, m2 = feat_mul.chunk(2, dim=1)

        # gate
        img_feat_gate = img_feat * feat_mul  # + img_feat
        modals_gate = modals * feat_mul  # + modals
        img_feat_gate = img_feat_gate + modals_gate
        # img_feat_gate = img_feat * m1 + modals * m2

        # fusion mask in feat and modals
        # then needs to be normalized
        if mask is not None and self.with_mask:
            mask = self.mask_convs(mask)
            img_feat_mask = img_feat * mask  # + img_feat
            modals_mask = modals * mask  # + modals
            img_feat = img_feat_gate + img_feat_mask + modals_mask
            # img_feat = img_feat_gate + img_feat_mask * m1 + modals_mask * m2
        else:
            img_feat = img_feat_gate

        # fusion feat and modals
        img_feat = self.fusion_proj(img_feat)
        # img_feat_modals = self.modals_out_ln(img_feat)
        img_feat_modals = img_feat

        ################################# LLM text and image attention ###########################
        # fusion image and llm features
        if llm_feat is not None and self.with_llm_feat:
            llm_feat = self.llm_drop(self.llm_dense(llm_feat))
            llm_feat = self.add_llm_pe(llm_feat)

            # downsample image
            img_feat, full_size = self.downsample_img_by_llm(img_feat)

            # ESS
            bs, C, H, W = img_feat.shape
            img_feat = self.scan(img_feat).view(bs * self.K, C, H * W).transpose(1, 2)
            img_feat = self.add_img_pe(img_feat, H, W)

            # multi-modal RWKV attention
            img_feat = self.multi_modal_img_ln(img_feat)
            llm_feat = self.multi_modal_txt_ln(llm_feat)
            # llm_feat: [bs, llm_len, c], img_feat: [bs, img_len, c]

            if self.add_mm_tokens:
                multi_modal_attn_out = self.multi_modal_attn(
                    img_feat,
                    llm_feat.repeat(self.K, 1, 1),
                    (H, W),
                    (self.img_token, self.txt_token),
                )
            else:
                multi_modal_attn_out = self.multi_modal_attn(
                    img_feat, llm_feat.repeat(self.K, 1, 1), (H, W)
                )

            # extract img and llm feat
            img_feat_out, llm_feat_out = self.extract_img_and_llm_feat(
                H, W, multi_modal_attn_out, self.with_llm_feat
            )

            # residual
            img_feat = img_feat + img_feat_out
            llm_feat = llm_feat + llm_feat_out.view(self.K, bs, -1, C).mean(dim=0)

            # merge ESS
            img_feat = rearrange(
                img_feat, "(b k) (h w) d -> b k d h w", k=self.K, h=H, w=W
            )
            img_feat = self.merge(img_feat) / self.K

            # upsample image
            img_feat = self.upsample_img_by_llm(img_feat, full_size)

            # mlps
            img_feat = self.img_mlp(img_feat) + img_feat
            if not self.last_enc_block:
                llm_feat = self.txt_mlp(llm_feat) + llm_feat

            # img_feat_llm = self.llm_out_ln(img_feat)
            img_feat_llm = img_feat

            return img_feat_modals, img_feat_llm, modals, llm_feat
        ###########################################################################################

        return img_feat_modals, modals, llm_feat

    def downsample_img_by_llm(self, img):
        full_size = img.shape[-2:]
        img_ds_size = (
            img.shape[-2] // self.img_ds_r_by_llm,
            img.shape[-1] // self.img_ds_r_by_llm,
        )
        # return F.interpolate(
        #     img,
        #     img_ds_size,
        #     mode='bilinear',
        #     align_corners=ALIGN_CORNERS,
        # ), full_size

        return interpolate("bilinear", img, img_sz=img_ds_size), full_size

    def upsample_img_by_llm(self, img, img_upsample_size: tuple[int, int]):
        # return F.interpolate(
        #     img,
        #     img_upsample_size,
        #     mode='bilinear',
        #     align_corners=ALIGN_CORNERS,
        # )
        return interpolate("bilinear", img, img_sz=img_upsample_size)

    def __repr__(self):
        return f"{self.__class__.__name__}(modal_chan={self.modal_chan}, \
            n_embd={self.n_embd}, mask_chan={self.mask_chan}, llm_chan={self.llm_chan})"


class DoubleStreamRWKVBlock(nn.Module):
    def __init__(
        self,
        layer_id,
        n_layer=8,
        n_embd=64,
        drop_path=0.0,
        scan_mode="K2",
        attn_bias=True,
        ffn_bias=True,
        cond_inj_mode="add",
        ffn_hidden_rate=1,
        *,
        no_out_norm=False,
        shift_type="conv",
        scan_id=0,
        window_size=0,
        no_roll=False,
        MIFM_modal_chan=1,
        MIFM_mask_chan=1,
        MIFM_llm_chan=None,
        MIFM_feat_drop=0.0,
        MIFM_llm_drop=0.0,
        MIFM_llm_pe_type="none",
        MIFM_img_pe_type="none",
        MIFM_ffn_type="llama",
        img_downsample_ratio_by_llm=4,
        add_mm_tokens=False,
        checkpoint=False,
        last_enc_block=False,
        block_residual=True,
    ):
        super().__init__()
        self.layer_id = layer_id
        self.drop_path = drop_path
        self.n_embd = n_embd
        self.no_out_norm = no_out_norm
        self.cond_inj_mode = cond_inj_mode
        self.window_size = window_size
        self.roll_size = (
            window_size // 2
            if (scan_id % 2 == 0 and window_size > 0 and not no_roll)
            else 0
        )
        self.img_downsample_ratio_by_llm = img_downsample_ratio_by_llm
        self.add_mm_tokens = add_mm_tokens
        self.last_enc_block = last_enc_block
        self.block_residual = block_residual

        # flags
        self.has_llm = MIFM_llm_chan is not None
        self.has_wind_part = window_size > 0
        self.has_roll = self.roll_size > 0

        # block config
        logger.info(
            f"DoubleStreamRWKVBlock: layer_id: {layer_id}, scan_mode: {scan_mode}, cond_inj_mode: {cond_inj_mode}, window_size: {window_size}, ",
            f"scan_id: {scan_id}, roll_size: {self.roll_size}, out_norm: {not no_out_norm}, ",
            f"MIFM_llm_pe_type: {MIFM_llm_pe_type}, MIFM_img_pe_type: {MIFM_img_pe_type}, add_mm_tokens: {add_mm_tokens}, shift_type: {shift_type},",
            f"block_residual: {block_residual}",
        )

        # if not no_out_norm:
        #     self.out_norm = RMSNorm(-1, n_embd, 3)

        K = int(scan_mode[1]) or 1

        if self.has_llm and not self.last_enc_block:
            self.txt_chan_reduce = nn.Linear(n_embd, n_embd, bias=False)

        ############# MIFM ##############
        self.MIFM = MultiModalityFusion(
            n_embd,
            MIFM_modal_chan,
            MIFM_mask_chan,
            MIFM_llm_chan,
            MIFM_feat_drop,
            MIFM_llm_drop,
            MIFM_llm_pe_type,
            MIFM_img_pe_type,
            scan_id=scan_id,
            layer_id=layer_id,
            img_ds_r_by_llm=img_downsample_ratio_by_llm,
            scan_mode=scan_mode,
            attn_bias=attn_bias,
            ffn_type=MIFM_ffn_type,
            shift_type=shift_type,
            n_layer=n_layer,
            add_mm_tokens=add_mm_tokens,
            last_enc_block=last_enc_block,
        )
        self.mm_proj_in = nn.Conv2d(n_embd * 3, n_embd, 1, bias=False)  # no bias
        # self.lerp_factor = nn.Parameter(torch.empty(n_embd).uniform_(-0.4, 0.4))
        self.lerp_factor = nn.Parameter(torch.zeros(n_embd))

        ############# BRWKV #############
        assert scan_mode in [
            "K0",
            "K2",
            "K4",
            "K8",
        ], "scan_mode should be one of [K0, K2, K4, K8]"
        self.K = K
        dim_att = n_embd
        dim_ffn = n_embd

        self.modulator1 = Modulator(dim_att, double=False, K=K)
        self.modulator2 = Modulator(dim_ffn, double=False, K=1)

        N_HEAD_ATTN = dim_att // HEAD_SIZE
        N_HEAD_FFN = dim_ffn // HEAD_SIZE

        # Attention
        self.att_img = RWKV_TMix(
            dim_att,
            N_HEAD_ATTN,
            n_layer,
            layer_id,
            shift_mode=shift_type,
            attn_bias=attn_bias,
            key_norm=ATTN_KEY_NORM,
            with_cp=checkpoint,
            scan_K=K,
            decay_speed=T_MIX_DECAY_SPEED,
        )

        # FFN
        self.ffn_fusion = RWKV_CMix(
            dim_ffn,
            N_HEAD_FFN,
            n_layer,
            layer_id,
            shift_mode=shift_type,
            hidden_rate=ffn_hidden_rate,
            ffn_bias=ffn_bias,
            key_norm=FFN_KEY_NORM,
            with_cp=checkpoint,
            scan_K=K,
        )
        if add_mm_tokens and self.has_llm:
            assert self.has_llm, "add_mm_tokens must be True when using llm feature"
            self.img_token = nn.Parameter(torch.zeros(2, dim_att, 1))
            self.txt_token = nn.Parameter(torch.zeros(2, dim_att, 1))

        self.only_img_attn_ln = norm_cls_channel_last(dim_att)
        self.fusion_img_llm_ln = norm_cls_channel_last(dim_ffn)
        # if self.has_llm:
        #     self.fusion_txt_ln = RMSNorm(-1, n_embd, 3)

        if drop_path > 0:
            self.drop_path = DropPath(drop_path)
        else:
            self.drop_path = nn.Identity()

        if scan_mode == "K0":
            self.scan = cross_scan_k0
            self.merge = cross_merge_k0
        elif scan_mode == "K2":
            self.scan = lambda x: CrossScanTritonSelect.apply(x, scan_id % 2)
            self.merge = lambda x: CrossMergeTritonSelect.apply(x, scan_id % 2)
        elif scan_mode == "K4":
            self.scan = CrossScanTriton.apply
            self.merge = CrossMergeTriton.apply
        else:
            self.scan = CrossScan.apply
            self.merge = CrossMerge.apply

    def partition_img_by_window(self, H, W, x):
        if self.has_wind_part:
            if self.has_roll:
                # roll_size = int(self.roll_size ** 2)
                # x = x.roll(shifts=-roll_size, dims=-1)
                x = x.roll(shifts=(-self.roll_size, -self.roll_size), dims=(-2, -1))
            # x = window_partition_1d(x, self.window_size)
            x = window_partition(x, self.window_size)
            # x = window_extract_2d(x, self.window_size)  # still holds in global receptive field
            h, w = self.window_size, self.window_size
        else:
            h, w = H, W

        return h, w, x

    def reverse_img_by_window(self, H, W, x, format="bchw"):
        if self.has_wind_part:
            x = window_reverse(x, self.window_size, H, W)
            if self.has_roll:
                x = x.roll(shifts=(self.roll_size, self.roll_size), dims=(-2, -1))

        # shape: [bs, h*w, c]
        return x

    def extract_img_and_llm_feat(
        self, h: int, w: int, att_or_ffn_out: torch.Tensor, has_llm: bool
    ):
        # att_or_ffn_out: [bs, K, C, L]
        if has_llm:
            if self.add_mm_tokens:
                img_feat, llm_feat = (
                    att_or_ffn_out[:, 1 : h * w + 1],
                    att_or_ffn_out[:, h * w + 3 : -1],
                )
            else:
                img_feat, llm_feat = (
                    att_or_ffn_out[:, : h * w],
                    att_or_ffn_out[:, h * w :],
                )
        else:
            img_feat = att_or_ffn_out
            llm_feat = None

        return img_feat, llm_feat

    def BRWKV_img_forward(
        self,
        x: torch.Tensor,
        MIMF_modals: torch.Tensor = None,
        MIFM_mm: torch.Tensor = None,
        llm_feat: torch.Tensor = None,
        patch_resolution=None,
    ):
        """
        ##! 1. Image-only attention with clipping windows
        w_img = window_partition(img, window_size)
        w_img = scan(img)
        w_img = only_img_attention(img)

        ##! 2. Multi-modal attention with txt feature
        # solution 1 (downsample image)
        img = window_reverse(w_img, window_size, H, W)
        ds_img = downsample_scanned_img(img)
        ds_img, txt = multi_modal_attention(ds_img, txt)
        upsample_img = upsample_scanned_img(ds_img)
        img = (upsample_img + img) / 2

        # solution 2 (repeat txt feature on bs-dim)
        rep_bs_txt = repeat(txt, 'bs d l -> (bs n_winds) d l')
        w_img, rep_bs_txt = multi_modal_attention(w_img, rep_bs_txt)
        img = window_reverse(w_img, window_size, H, W)
        txt = rearrange(rep_bs_txt, '(bs n_winds) d l -> bs n_winds d l', bs=B)
        txt = txt.mean(dim=1)

        ##! 3. Multi-modal FFN
        #! Flux.1 says we should have two FFNs, one for image and one for txt
        img = FFN(img, txt)

        """

        B, C, H, W = x.shape
        # has_llm = (llm_feat is not None) and self.has_llm

        # pre-fusion previous (non-downsampled) image and MIFM image
        # x = x + MIFM_img

        # ver 3
        # MIFM_mm = MIFM_mm * self.lerp_factor.view(1, -1, 1, 1)
        # x = self.mm_proj_in(torch.cat([x, MIFM_mm], dim=1))
        x = self.mm_proj_in(
            torch.cat(
                [x, MIFM_mm * self.lerp_factor.view(1, -1, 1, 1), MIMF_modals], dim=1
            )
        )

        # we perform scanning first and then window partition
        # different from v11 where we perform window partition first

        # window partition
        # (bs, c, h * w) -> (bs x n_winds, c, window_size, window_size)
        h, w, x = self.partition_img_by_window(H, W, x)

        # ESS
        x = (
            self.scan(x).view(x.size(0) * self.K, -1, h * w).transpose(1, 2)
        )  # [b*k, l, c]

        ######################## Image-only Attention ##########################

        # Image spatial mixing
        x = self.only_img_attn_ln(x)
        sc1, sh1, ga1 = self.modulator1(x)
        prenorm_x = (1 + sc1) * x + sh1  # modulated img
        x_attn = self.drop_path(self.att_img(prenorm_x, None, (h, w)))
        x = ga1 * x_attn + x

        # merge ESS
        # (bs, c * K, H * W) -> (bs, c, H, W)
        x = rearrange(x, "(b k) (h w) d -> b k d h w", k=self.K, h=h, w=w)
        x = self.merge(x) / self.K

        # window reverse
        # (bs x n_winds, c, wind x wind) -> (bs, c, h x w)
        x = self.reverse_img_by_window(H, W, x)
        x = x.view(B, C, H * W).transpose(1, 2)

        ########################################################################

        ######################## Image-modality FFN ############################

        # Channel mixing
        x = self.fusion_img_llm_ln(x)
        sc2, sh2, ga2 = self.modulator2(x)
        prenorm_x = (1 + sc2) * x + sh2

        # 2. pure image FFN
        x_ffn = self.drop_path(self.ffn_fusion(prenorm_x, None, (H, W)))
        x = ga2 * x_ffn + x

        x = x.transpose(1, 2).view(B, C, H, W)

        #########################################################################

        # txt reduced channel
        if (not self.last_enc_block) and self.has_llm and llm_feat is not None:
            llm_feat = self.txt_chan_reduce(llm_feat)

        return x, llm_feat

    def forward(
        self,
        x: torch.Tensor,
        cond: ConditionInput,
        patch_resolution: tuple[int, int] | None = None,
    ):
        inp = x
        ## MIFM forward
        MIFM_modals_feat, MIFM_mm_feat, modals, llm_feat = self.MIFM(x, cond)

        ## BRWKV forward
        x, llm_feat = self.BRWKV_img_forward(
            x, MIFM_modals_feat, MIFM_mm_feat, llm_feat, patch_resolution
        )

        # add residual
        if self.block_residual:
            x = x + inp

        return x, ConditionInput(cond.modalities, llm_feat, cond.mask_input)


class SingleStreamRWKVBlock(nn.Module):
    def __init__(
        self,
        layer_id,
        n_layer=8,
        n_embd=64,
        drop_path=0.0,
        scan_mode="K2",
        attn_bias=True,
        ffn_bias=True,
        ffn_hidden_rate=1,
        *,
        no_out_norm=False,
        shift_type="conv",
        scan_id=0,
        window_size=0,
        no_roll=False,
        MIFM_modal_chan=1,
        MIFM_feat_drop=0.0,
        checkpoint=False,
        block_residual=False,
    ):
        super().__init__()
        self.layer_id = layer_id
        self.drop_path = drop_path
        self.n_embd = n_embd
        self.no_out_norm = no_out_norm
        self.window_size = window_size
        self.roll_size = (
            window_size // 2
            if (scan_id % 2 == 0 and window_size > 0 and not no_roll)
            else 0
        )
        self.block_residual = block_residual

        # block config
        logger.info(
            f"SingleStreamRWKVBlock: layer_id: {layer_id}, scan_mode: {scan_mode}, window_size: {window_size}, ",
            f"scan_id: {scan_id}, roll_size: {self.roll_size}, out_norm: {not no_out_norm},",
            f"shift_type: {shift_type}, block_residual: {block_residual}",
        )

        # if not no_out_norm:
        #     self.out_norm = RMSNorm(1, n_embd, 4)

        ############# MIFM ##############
        # only use the modal as condition
        self.MIFM = MultiModalityFusion(
            n_embd, MIFM_modal_chan, feat_drop=MIFM_feat_drop
        )
        self.mm_proj_in = nn.Conv2d(n_embd * 2, n_embd, 1, bias=False)  # no bias
        # self.lerp_factor = nn.Parameter(torch.zeros(n_embd))

        ############# BRWKV #############
        assert scan_mode in [
            "K0",
            "K2",
            "K4",
            "K8",
        ], "scan_mode should be one of [K0, K2, K4, K8]"
        K = int(scan_mode[1]) or 1
        self.K = K
        dim_att = n_embd
        dim_ffn = n_embd

        self.attn_modulator = Modulator(dim_att, double=False)
        self.ffn_modulator = Modulator(dim_ffn, double=False)

        N_HEAD_ATTN = dim_att // HEAD_SIZE
        N_HEAD_FFN = dim_ffn // HEAD_SIZE

        self.att = RWKV_TMix(
            dim_att,
            N_HEAD_ATTN,
            n_layer,
            layer_id,
            shift_mode=shift_type,
            attn_bias=attn_bias,
            key_norm=ATTN_KEY_NORM,
            with_cp=checkpoint,
            scan_K=K,
            decay_speed=T_MIX_DECAY_SPEED,
        )
        self.ffn = RWKV_CMix(
            dim_ffn,
            N_HEAD_FFN,
            n_layer,
            layer_id,
            shift_mode=shift_type,
            hidden_rate=ffn_hidden_rate,
            ffn_bias=ffn_bias,
            key_norm=FFN_KEY_NORM,
            with_cp=checkpoint,
            scan_K=K,
        )

        self.x_ln1 = norm_cls_channel_last(dim_att)
        self.x_ln2 = norm_cls_channel_last(dim_ffn)

        if drop_path > 0:
            self.drop_path = DropPath(drop_path)
        else:
            self.drop_path = nn.Identity()

        if scan_mode == "K0":
            self.scan = cross_scan_k0
            self.merge = cross_merge_k0
        elif scan_mode == "K2":
            self.scan = lambda x: CrossScanTritonSelect.apply(x, scan_id % 2)
            self.merge = lambda x: CrossMergeTritonSelect.apply(x, scan_id % 2)
        elif scan_mode == "K4":
            self.scan = CrossScanTriton.apply
            self.merge = CrossMergeTriton.apply
        else:
            self.scan = CrossScan.apply
            self.merge = CrossMerge.apply

    def forward(
        self,
        x: torch.Tensor,
        cond: ConditionInput,
        patch_resolution: tuple | None = None,
    ):
        B, C, H, W = x.shape
        inp = x

        ## MIFM
        MIFM_modals, *_ = self.MIFM(x, cond)
        x = self.mm_proj_in(torch.cat([x, MIFM_modals], dim=1))

        ## BRWKV
        # window partition
        if self.window_size > 0:
            if self.roll_size > 0:
                x = x.roll(shifts=(-self.roll_size, -self.roll_size), dims=(-2, -1))
            x = window_partition(x, self.window_size)
            h, w = self.window_size, self.window_size
        else:
            h, w = H, W

        # ESS
        x = self.scan(x).view(x.size(0) * self.K, -1, h * w).transpose(1, 2)

        # Spatial mixing
        x = self.x_ln1(x)
        sc1, sh1, ga1 = self.attn_modulator(x)
        prenorm_x = (1 + sc1) * x + sh1

        x = ga1 * self.drop_path(self.att(prenorm_x, None, (h, w))) + x

        # merge ESS
        x = rearrange(x, "(b k) (h w) d -> b k d h w", k=self.K, h=h, w=w)
        x = self.merge(x) / self.K

        # window reverse
        if self.window_size > 0:
            x = window_reverse(x, self.window_size, H, W)
            if self.roll_size > 0:
                x = x.roll(shifts=(self.roll_size, self.roll_size), dims=(-2, -1))
        x = x.view(B, C, H * W).transpose(1, 2)

        # Channel mixing
        x = self.x_ln2(x)
        sc2, sh2, ga2 = self.ffn_modulator(x)
        prenorm_x = (1 + sc2) * x + sh2

        x = ga2 * self.drop_path(self.ffn(prenorm_x, None, (H, W))) + x

        x = x.transpose(1, 2).view(B, C, H, W)

        # add residual
        if self.block_residual:
            x = x + inp

        return x


class FusionSequential(nn.Module):
    def __init__(self, *args, dim: int = None, resi_connection: str = "3conv"):
        super().__init__()
        self.mods = nn.ModuleList(args)
        self.resi_connection = resi_connection

        if resi_connection == "1conv":
            self.conv = nn.Conv2d(dim, dim, 3, 1, 1, padding_mode=CONV_PAD_MODE)
        elif resi_connection == "3conv":
            # to save parameters and memory
            r_dim = dim // 2
            self.conv = nn.Sequential(
                nn.Conv2d(dim, r_dim, 3, 1, 1, padding_mode=CONV_PAD_MODE),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(r_dim, r_dim, 1, 1, 0, padding_mode=CONV_PAD_MODE),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(r_dim, dim, 3, 1, 1, padding_mode=CONV_PAD_MODE),
            )
        else:
            # logger.warning(f"resi_connection {resi_connection} not add")
            raise NotImplementedError(
                f"resi_connection {resi_connection} not implemented"
            )

        # print info in this layer
        logger.info(f"resi_connection {resi_connection} added in one RWKVFusion layer")
        logger.info(
            "====================================================================="
        )

    def __getitem__(self, idx):
        return self.mods[idx]

    def patch_unembed(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        # x: [B, L, C]
        assert x.ndim == 3, "x should be a 3D tensor"
        assert x.shape[1] == H * W, "x should have the same length as the image"

        return rearrange(x, "b (h w) c -> b c h w", h=H, w=W)

    def patch_embed(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W]
        assert x.ndim == 4, "x should be a 4D tensor"

        return rearrange(x, "b c h w -> b (h w) c")

    def enc_forward(
        self, feat, cond, patch_resolution
    ) -> tuple[torch.Tensor, ConditionInput]:
        outp = feat
        for i, mod in enumerate(self.mods):
            outp, cond = mod(outp, cond, patch_resolution)
        if self.resi_connection:
            # swinir type conv residual connection
            outp = outp.contiguous()
            outp = self.conv(outp)
            outp = outp + feat
        else:
            outp = outp.contiguous()

        return outp, cond

    def dec_forward(self, feat, cond, patch_resolution) -> torch.Tensor:
        outp = self.mods[0](feat)  # to U-Net encoder conv

        feat = outp
        for i, mod in enumerate(self.mods[1:]):
            outp = mod(outp, cond, patch_resolution)
        if self.resi_connection:
            # swinir type conv residual connection
            outp = outp.contiguous()
            outp = self.conv(outp)
            outp = outp + feat
        else:
            outp = outp.contiguous()

        return outp


# *==============================================================
# * RWKVFusion Model
# *==============================================================


class UNetDownsample(nn.Module):
    def __init__(self, chan, r=2, chan_r=2, down_type="conv"):
        super().__init__()
        self.norm_scale = nn.Parameter(
            torch.full((chan * chan_r,), r).sqrt()
        )  # amplify the norm scale

        if down_type == "conv":
            self.down = nn.Conv2d(
                chan, chan * chan_r, r, r, bias=False, padding_mode=CONV_PAD_MODE
            )
        elif down_type == "pixelunshuffle":
            # dim -> (conv) -> dim * chan_r -> (pixel_unshuffle) -> dim * chan_r * r ** 2
            dim_out = chan * chan_r
            assert dim_out % (r**2) == 0, "dim_out should be divisible by r ** 2"
            dim_mid = dim_out // (r**2)
            self.down = nn.Sequential(
                nn.Conv2d(
                    chan, dim_mid, 3, 1, 1, bias=False, padding_mode=CONV_PAD_MODE
                ),
                nn.PixelUnshuffle(r),
            )
        else:
            raise NotImplementedError(f"down type {down_type} not implemented")

        logger.info(f"downsample type: {down_type}")

    def forward(self, x):
        # keep norm
        return self.down(x) * self.norm_scale.view(1, -1, 1, 1)


class UNetUpsample(nn.Module):
    def __init__(
        self, chan, r=2, chan_r=2, up_type="upsample_bilinear", antialias=False
    ):
        super().__init__()
        self.up_type = up_type.split("_")[0]
        self.antialias = antialias
        self.norm_scale = nn.Parameter(
            torch.full((chan // chan_r,), r).rsqrt()
        )  # reduce the norm scale
        self.r = r
        self.resample_mode = (
            up_type.split("_")[1] if self.up_type == "upsample" else None
        )

        if self.up_type == "upsample":
            self.up_conv = nn.Conv2d(chan, chan // chan_r, 1, bias=False)
            # self._align_corners = ALIGN_CORNERS if self.resample_mode != 'nearest' else None
            # self._antialias = self.antialias if self.resample_mode != 'nearest' else None
        elif self.up_type == "transpose":
            self.up = nn.Sequential(
                nn.Conv2d(chan, chan // chan_r, 1, bias=False),
                nn.ConvTranspose2d(
                    chan // chan_r,
                    chan // chan_r,
                    2,
                    2,
                    bias=False,
                    groups=chan // chan_r,
                ),
            )
        elif self.up_type == "pixelshuffle":
            dim_out = chan // chan_r
            assert dim_out % (r**2) == 0, "dim_out should be divisible by r ** 2"
            dim_mid = dim_out * (r**2)
            self.up = nn.Sequential(
                nn.Conv2d(
                    chan, dim_mid, 3, 1, 1, bias=False, padding_mode=CONV_PAD_MODE
                ),
                nn.PixelShuffle(r),
            )
        else:
            raise NotImplementedError(f"up type {up_type} not implemented")

        logger.info(f"upsample type: {up_type}")

    def forward(self, x):
        if self.up_type == "upsample":
            x = self.up_conv(x) * self.norm_scale.view(1, -1, 1, 1)
            # return F.interpolate(x,
            #                      scale_factor=self.r,
            #                      mode=self.resample_mode,
            #                      align_corners=self._align_corners,
            #                      antialias=self._antialias)
            return interpolate(
                self.resample_mode,
                x,
                factor=self.r,
            )
        elif self.up_type == "transpose":
            return self.up(x) * self.norm_scale.view(1, -1, 1, 1)
        elif self.up_type == "pixelshuffle":
            return self.up(x) * self.norm_scale.view(1, -1, 1, 1)
        else:
            raise NotImplementedError(f"up type {self.up_type} not implemented")


@register_model("RWKVFusion")
class RWKVFusion(BaseModel):
    @typechecked
    def __init__(
        self,
        ############### Basic Config ###############
        img_channel: int = 3,
        modal_channel: int = 1,
        mask_channel: int | None = None,
        llm_channel: int | None = None,
        out_channel: int = 3,
        width: int = 16,
        ############### U-Net Config ###############
        middle_blk_num: int = 1,
        mid_window_size: int = 0,
        enc_blk_nums: list = [],
        dec_blk_nums: list = [],
        chan_upscales: list = [],
        enc_window_sizes: list = [],
        dec_window_sizes: list = [],
        ffn_hidden_rates: list = [],
        scan_mode: str = "K2",
        with_checkpoint: bool = False,
        drop_path_rate: float = 0.1,
        no_out_norm: bool = True,
        layer_resi_connection: str = "3conv",
        add_block_residual: bool = False,
        shift_type: str = "none",
        ############### Fusion Config ###############
        upscale: int = 1,
        if_abs_pos: bool = False,
        pt_img_size: int | Sequence[int] = 64,
        fusion_prior: Literal["max", "mean", "lr", "grad", "none"] = "lr",
        end_proj_type: str = "3conv",
        patch_merge: bool = True,
        ############### Multi-modal Config ###############
        drop_txt_ratio: float = 0.0,
        txt_embd_drop_ratio: float = 0.0,
        multi_value_mask_max_classes: int | None = None,
        max_txt_length: int = 512,
        add_multi_modal_tokens: bool = False,
        MIFM_ffn_type: Literal["llama", "qwen2"] = "qwen2",
        MIFM_img_downsample_ratios: list | None = None,
        ############### Feature Prior Config ###############
        feature_prior: bool = False,
        reconstruction_head: str = "sr",
    ):
        super().__init__()
        self.upscale = upscale
        self.if_abs_pos = if_abs_pos
        pt_img_size = (
            list(pt_img_size)
            if isinstance(pt_img_size, (tuple, list))
            else [pt_img_size, pt_img_size]
        )
        self.pt_img_size = pt_img_size
        self.fusion_prior = fusion_prior
        self.patch_merge = patch_merge
        self.mask_channel = mask_channel
        self.drop_txt_ratio = drop_txt_ratio
        self.multi_value_mask_max_classes = multi_value_mask_max_classes
        self.feature_prior = feature_prior
        self.reconstruction_head = reconstruction_head

        # flags
        self.has_mask = (
            mask_channel is not None or multi_value_mask_max_classes is not None
        )
        self.has_txt = llm_channel is not None

        # type check
        assert (
            len(enc_blk_nums)
            == len(dec_blk_nums)
            == len(chan_upscales)
            == len(enc_window_sizes)
            == len(dec_window_sizes)
            == len(ffn_hidden_rates)
            == len(MIFM_img_downsample_ratios)
        ), "config length not match"
        assert fusion_prior in [
            "max",
            "mean",
            "grad",
            "lr",
            "none",
        ], "`fusion_prior` should be one of [max, mean, grad, lr, none]"

        # cfgs
        net_name = __class__.__name__
        logger.info(
            f"{net_name}: multi-modal settings, input mask: {self.has_mask}, input txt: {self.has_txt}\n"
            f"{net_name}: enc_blk_nums: {enc_blk_nums}, dec_blk_nums: {dec_blk_nums}\n"
            f"{net_name}: chan_upscales: {chan_upscales}, enc_window_sizes: {enc_window_sizes}, dec_window_sizes: {dec_window_sizes}\n"
            f"{net_name}: checkpoint: {with_checkpoint} {'the training will be slow for large batch size' if with_checkpoint else ''}\n"
            f"{net_name}: fusion_prior: {fusion_prior}, use patch merge: {patch_merge}\n"
            f"{net_name}: txt input directly drop ratio: {drop_txt_ratio}, txt embd drop ratio: {txt_embd_drop_ratio}"
            f"{__class__.__name__}: {img_channel=}, {modal_channel=}\n"
        )

        if if_abs_pos:
            self.abs_pos = nn.Parameter(
                torch.randn(1, 1, pt_img_size[0], pt_img_size[1]), requires_grad=True
            )

        self.patch_embd = nn.Conv2d(
            in_channels=img_channel + modal_channel,
            out_channels=width,
            kernel_size=1,
            stride=1,
            bias=False,
        )
        modal_channel_in = img_channel + modal_channel
        modal_channel = width
        self.modal_in = nn.Conv2d(modal_channel_in, modal_channel, 1, 1, 0, bias=False)

        # * --- Multi-modal Conditions Embedding --- #

        ## llm embedding
        if llm_channel is not None:
            self.llm_channel = width * 2
            self.llm_embd = nn.Sequential(
                # nn.LayerNorm(llm_channel),
                nn.Linear(llm_channel, self.llm_channel, bias=False),
                nn.Linear(self.llm_channel, self.llm_channel, bias=False),
            )
            self.embd_drop = nn.Dropout(txt_embd_drop_ratio)
            txt_ids = np.arange(max_txt_length)
            txt_pe = torch.from_numpy(
                get_1d_sincos_pos_embed_from_grid(self.llm_channel, txt_ids)
            )  # [l, d]
            self.register_buffer("txt_sine_pe", txt_pe)
            llm_channel = self.llm_channel

        ## mask embedding
        if self.multi_value_mask_max_classes is not None:
            self.mask_embd = nn.Conv2d(
                self.multi_value_mask_max_classes, width, 1, bias=False
            )
            mask_channel = width

        # * --- Reconstruction Head --- #

        ## convs after body
        if end_proj_type == "1conv":
            self.conv_after_body = nn.Conv2d(
                in_channels=width,
                out_channels=width,
                kernel_size=1,
                stride=1,
                bias=False,
            )
        elif end_proj_type == "3conv":
            bn_chan = width // 2
            self.conv_after_body = nn.Sequential(
                nn.Conv2d(
                    width, bn_chan, 3, 1, 1, bias=False, padding_mode=CONV_PAD_MODE
                ),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(
                    bn_chan, bn_chan, 1, 1, 0, bias=False, padding_mode=CONV_PAD_MODE
                ),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(
                    bn_chan, width, 3, 1, 1, bias=False, padding_mode=CONV_PAD_MODE
                ),
            )
        else:
            raise NotImplementedError(f"end_proj_type {end_proj_type} not implemented")

        ## fusion prior convs
        if self.fusion_prior != "none" and self.feature_prior:
            self.prior_convs = nn.Sequential(
                nn.Conv2d(out_channel, width, 1, 1, bias=False),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(
                    width,
                    width,
                    3,
                    1,
                    1,
                    groups=width,
                    bias=False,
                    padding_mode=CONV_PAD_MODE,
                ),
            )

        ## fusion head
        self.fusion_head = nn.Conv2d(
            width, out_channel, 3, 1, 1, bias=True, padding_mode=CONV_PAD_MODE
        )

        # * --- Main body --- #

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()

        depth = sum(enc_blk_nums) + middle_blk_num + sum(dec_blk_nums)
        # stochastic depth decay rule
        inter_dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth // 2)]
        inter_dpr = inter_dpr + inter_dpr[::-1]
        if len(inter_dpr) < depth:
            inter_dpr.append(inter_dpr[-1])
        logger.info(
            f"{net_name}: inter_dpr: {np.array(inter_dpr).round(4)} for depth: {depth}"
        )

        chan = width
        n_prev_blks = 0
        n_prev_layers = 0
        # encoder
        for layer_id, num in enumerate(enc_blk_nums):
            encoder_seq = []
            for i in range(num):
                encoder_seq.append(
                    DoubleStreamRWKVBlock(
                        layer_id=n_prev_blks,
                        n_layer=depth,
                        n_embd=chan,
                        window_size=enc_window_sizes[layer_id],
                        drop_path=inter_dpr[n_prev_blks],
                        scan_id=n_prev_blks,  # n_prev_layers + layer_id,
                        ffn_hidden_rate=ffn_hidden_rates[layer_id],
                        MIFM_modal_chan=modal_channel,
                        MIFM_mask_chan=mask_channel,
                        MIFM_llm_chan=(llm_channel if n_prev_blks == 0 else prev_chan)
                        if self.has_txt
                        else None,
                        MIFM_feat_drop=inter_dpr[n_prev_blks],
                        MIFM_llm_drop=inter_dpr[n_prev_blks],
                        MIFM_llm_pe_type="none",
                        MIFM_img_pe_type="none",
                        img_downsample_ratio_by_llm=MIFM_img_downsample_ratios[
                            layer_id
                        ],
                        add_mm_tokens=add_multi_modal_tokens,
                        scan_mode=scan_mode,
                        checkpoint=with_checkpoint,
                        no_out_norm=no_out_norm,
                        MIFM_ffn_type=MIFM_ffn_type,
                        shift_type=shift_type,
                        no_roll=False,
                        block_residual=add_block_residual,
                    )
                )
                n_prev_blks += 1
                prev_chan = chan
            self.encoders.append(
                FusionSequential(
                    *encoder_seq, dim=chan, resi_connection=layer_resi_connection
                )
            )
            self.downs.append(
                UNetDownsample(
                    chan, r=2, chan_r=chan_upscales[layer_id], down_type="conv"
                )
            )
            chan = chan * chan_upscales[layer_id]
            pt_img_size[0] //= 2
            pt_img_size[1] //= 2
        n_prev_layers += len(enc_blk_nums)

        # middle layer
        self.middle_blks = FusionSequential(
            *[
                DoubleStreamRWKVBlock(
                    layer_id=n_prev_blks + i,
                    n_layer=depth,
                    n_embd=chan,
                    window_size=mid_window_size,
                    drop_path=inter_dpr[n_prev_blks + i],
                    scan_id=n_prev_blks + i,  # n_prev_layers + layer_id,
                    ffn_hidden_rate=ffn_hidden_rates[layer_id],
                    MIFM_modal_chan=modal_channel,
                    MIFM_mask_chan=mask_channel,
                    MIFM_llm_chan=prev_chan if self.has_txt else None,
                    MIFM_feat_drop=inter_dpr[n_prev_blks + i],
                    MIFM_llm_drop=inter_dpr[n_prev_blks + i],
                    MIFM_llm_pe_type="none",
                    MIFM_img_pe_type="none",
                    img_downsample_ratio_by_llm=MIFM_img_downsample_ratios[-1],
                    add_mm_tokens=add_multi_modal_tokens,
                    checkpoint=with_checkpoint,
                    scan_mode=scan_mode,
                    last_enc_block=True if i == middle_blk_num - 1 else False,
                    no_out_norm=no_out_norm,
                    MIFM_ffn_type=MIFM_ffn_type,
                    shift_type=shift_type,
                    no_roll=False,
                    block_residual=add_block_residual,
                )
                for i in range(middle_blk_num)
            ],
            dim=chan,
            resi_connection=layer_resi_connection,
        )
        n_prev_blks += middle_blk_num
        n_prev_layers += 1

        # decoder
        self.skip_scales = nn.ParameterList([])
        for dec_layer_id, num in enumerate(reversed(dec_blk_nums)):
            # self.ups.append(up(chan, r=2, chan_r=chan_upscales[::-1][dec_layer_id]))
            self.ups.append(
                UNetUpsample(
                    chan,
                    r=2,
                    chan_r=chan_upscales[::-1][dec_layer_id],
                    up_type="upsample_bilinear",
                )
            )
            chan = chan // chan_upscales[::-1][dec_layer_id]
            window_size = dec_window_sizes[::-1][dec_layer_id]
            ffn_hidden_rate = ffn_hidden_rates[::-1][dec_layer_id]
            pt_img_size[0] *= 2
            pt_img_size[1] *= 2
            last_layer = dec_layer_id == len(dec_blk_nums) - 1
            self.skip_scales.append(
                nn.Parameter(torch.full((chan,), 2).rsqrt(), requires_grad=True)
            )

            self.decoders.append(
                FusionSequential(
                    nn.Conv2d(chan * 2, chan, 1, bias=False),
                    *[
                        SingleStreamRWKVBlock(
                            layer_id=n_prev_blks + i,
                            n_layer=depth,
                            n_embd=chan,
                            window_size=window_size,
                            drop_path=inter_dpr[n_prev_blks + i],
                            scan_id=n_prev_blks + i,
                            ffn_hidden_rate=ffn_hidden_rate,
                            MIFM_modal_chan=modal_channel,
                            MIFM_feat_drop=inter_dpr[n_prev_blks + i],
                            checkpoint=with_checkpoint,
                            no_out_norm=no_out_norm,
                            scan_mode=scan_mode,
                            shift_type=shift_type,
                            no_roll=False,  # no roll for decoder
                            block_residual=add_block_residual,
                        )
                        for i in range(num)
                    ],
                    dim=chan,
                    resi_connection=layer_resi_connection,
                )
            )
            n_prev_blks += num
        n_prev_layers += len(dec_blk_nums)

        # Other functions

        # self.interpolate = lambda x, H, W: F.interpolate(x, (H, W), mode='bilinear', align_corners=ALIGN_CORNERS)
        # self.interpolate = lambda x, H, W: interpolate('bilinear', x, img_sz=(H, W))

    def _init_weights(self, m: nn.Module):
        rng_linear = torch.Generator().manual_seed(2025)
        rng_conv = torch.Generator().manual_seed(2026)

        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02, generator=rng_linear)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.weight, 1.0)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, LayerNorm):
            nn.init.constant_(m.body.weight, 1.0)
            if hasattr(m.body, "bias"):
                nn.init.constant_(m.body.bias, 0)
        elif isinstance(m, (nn.Conv1d, nn.Conv2d)):
            # trunc_normal_(m.weight, std=0.02, generator=rng_conv)
            nn.init.xavier_normal_(m.weight, gain=1.0, generator=rng_conv)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def resize_pos_embd(self, pos_embd: torch.Tensor, inp_size: tuple):
        if pos_embd.size(-1) != inp_size[-1] or pos_embd.size(-2) != inp_size[-2]:
            pos_embd = F.interpolate(pos_embd, inp_size[-2:], mode="bilinear")

        return pos_embd

    def resize_conds(
        self, H, W, x: torch.Tensor = None, resample_type: str = "bilinear"
    ):
        if exists(x):
            # x = self.interpolate(x, H, W)
            x = interpolate(resample_type, x, img_sz=(H, W))

        return x

    def _forward_features(self, inp, modal, mask=None, llm_feature=None):
        bs, _, H, W = inp.shape

        cat_x_modal = torch.cat([inp, modal], dim=1)
        patch_embd_x = self.patch_embd(cat_x_modal)
        modal = self.modal_in(cat_x_modal)

        if self.if_abs_pos:
            x = patch_embd_x + self.resize_pos_embd(
                self.abs_pos, inp_size=(H, W)
            ).expand(bs, patch_embd_x.size(1), -1, -1)
        else:
            x = patch_embd_x

        if llm_feature is not None and self.llm_channel is not None:
            llm_feature = F.normalize(llm_feature, p=2, dim=1)
            llm_feature = self.llm_embd(llm_feature)
            llm_feature = (llm_feature + self.txt_sine_pe).float()
            llm_feature = self.embd_drop(llm_feature)

        if mask is not None and hasattr(self, "mask_embd"):
            mask = self.mask_embd(mask)

        # init condition
        u_cond = ConditionInput(modal, llm_feature, mask)

        ## encoder
        encs = []
        for encoder, down in zip(self.encoders, self.downs):
            u_cond.modalities = self.resize_conds(
                H, W, u_cond.modalities, resample_type="bilinear"
            )
            u_cond.mask_input = self.resize_conds(
                H, W, u_cond.mask_input, resample_type="nearest"
            )
            x, u_cond = encoder.enc_forward(x, u_cond, (H, W))
            encs.append(x)
            x = down(x)
            H = H // 2
            W = W // 2

        ## middle layer
        u_cond.modalities = self.resize_conds(
            H, W, u_cond.modalities, resample_type="bilinear"
        )
        u_cond.mask_input = self.resize_conds(
            H, W, u_cond.mask_input, resample_type="nearest"
        )
        x, _ = self.middle_blks.enc_forward(x, u_cond, (H, W))

        ## decoder
        for decoder, up, enc_skip, skip_scale in zip(
            self.decoders, self.ups, encs[::-1], self.skip_scales
        ):
            x = up(x)
            H = H * 2
            W = W * 2
            u_cond.modalities = self.resize_conds(
                H, W, u_cond.modalities, resample_type="bilinear"
            )
            u_cond.mask_input = self.resize_conds(
                H, W, u_cond.mask_input, resample_type="nearest"
            )
            x = torch.cat([x, enc_skip * skip_scale.view(1, -1, 1, 1)], dim=1)
            x = decoder.dec_forward(x, u_cond, (H, W))

        return x, patch_embd_x

    def _foward_prior(self, inp, modal):
        # fusion prior
        if self.fusion_prior == "max":
            prior = torch.max(inp, modal)
            if self.feature_prior:
                prior = self.prior_convs(prior)
        elif self.fusion_prior == "mean":
            prior = (inp + modal) / 2.0
        elif self.fusion_prior == "lr":
            prior = inp
            if self.feature_prior:
                prior = self.prior_convs(prior)
        elif self.fusion_prior.startswith("grad"):
            inp_grad = sobel_op(inp, normalized=True)
            modal_grad = sobel_op(modal, normalized=True)
            prior = inp_grad.maximum(modal_grad)
        elif self.fusion_prior == "none":
            prior = 0.0
        else:
            raise ValueError(f"Invalid fusion_prior: {self.fusion_prior}")

        # if self.feature_prior:
        #     prior = self.prior_convs(prior)

        return prior

    def _forward_recon(self, feature_out, prior, patch_embd_x=None):
        if self.reconstruction_head == "sr":
            feature_out = self.conv_after_body(feature_out)
            if self.feature_prior:
                return self.fusion_head(feature_out + prior)
            else:
                return self.fusion_head(feature_out) + prior
        elif self.reconstruction_head == "fusion":
            assert (
                patch_embd_x is not None
            ), "patch_embd_x should be provided for fusion reconstruction"
            feature_out = self.conv_after_body(feature_out) + patch_embd_x
            if self.feature_prior:
                return self.fusion_head(feature_out + prior)
            else:
                return self.fusion_head(feature_out) + prior
        else:
            raise ValueError(f"Invalid reconstruction_head: {self.reconstruction_head}")

    def _forward_implem(self, inp, modal, mask=None, llm_feature=None):
        # inp: vi/over/far/SPECT/LRMS
        # modal: ir/under/near/MRI/PAN

        # forward features
        feature_out, patch_embd_x = self._forward_features(
            inp, modal, mask, llm_feature
        )

        # forward prior
        prior = self._foward_prior(inp, modal)

        # reconstruction
        outp = self._forward_recon(feature_out, prior, patch_embd_x)

        return outp

    def drop_txt(self, txt):
        if self.has_txt and txt is not None and self.drop_txt_ratio > 0:
            if np.random.rand() < self.drop_txt_ratio:
                txt = torch.zeros_like(txt)

        return txt

    def sharpening_train_step(
        self,
        lms: torch.Tensor,
        pan: torch.Tensor,
        gt: torch.Tensor,
        txt: torch.Tensor | None = None,
        criterion: "Callable | None" = None,
        **_kwargs,
    ):
        assert criterion is not None, "criterion should be provided"

        txt = self.drop_txt(txt)
        sr = self._forward_implem(lms, pan, None, txt)
        sr = sr.clip(0, 1)
        loss = criterion(sr, gt)

        return sr, loss

    @torch.no_grad()
    def sharpening_val_step(
        self,
        lms: torch.Tensor,
        pan: torch.Tensor,
        txt: "torch.Tensor | None" = None,
        patch_merge: "callable | bool | None" = True,
        *,
        inference_wo_txt: bool = False,
        **_kwargs,
    ):
        if patch_merge is None:
            patch_merge = self.patch_merge

        if inference_wo_txt and txt is not None and self.has_txt:
            txt = torch.zeros_like(txt)

        if patch_merge:
            logger.debug(f"using patch merge module")
            # for any patch merge, we drop the txt feature
            if txt is not None and self.has_txt:
                pm_step = partial(self.patch_merge_step, llm_feature=txt)
            else:
                pm_step = self.patch_merge_step

            _patch_merge_model = PatchMergeModule(
                self,
                crop_batch_size=64,
                patch_size_list=[16 * self.upscale, 16 * self.upscale],
                scale=1,
                patch_merge_step=pm_step,
            )
            sr = _patch_merge_model.forward_chop(lms, pan)[0]
        else:
            sr = self._forward_implem(lms, pan, None, txt)

        return sr.clip(0, 1)

    def only_fusion_step(self, vi, ir, mask=None, txt=None, **_kwargs):
        outp = self._forward_implem(vi, ir, mask, txt)

        return outp.clip(0, 1)

    def check_multi_modal_inputs(
        self,
        vis: torch.Tensor,
        ir: torch.Tensor,
        mask: torch.Tensor | None,
        txt: torch.Tensor | None,
        device: "torch.device | str",
        dtype: torch.dtype,
    ):
        if not self.has_mask:
            mask = None
        elif mask is not None:
            mask = mask.to(device).type(dtype)
        else:  # has mask in model but input mask is None
            mask = torch.zeros(
                1, self.mask_channel, *vis.shape[-2:], device=device, dtype=dtype
            )

        if not self.has_txt:
            txt = None
        elif txt is not None:
            txt = txt.to(device).type(dtype)

        return vis, ir, mask, txt

    def check_multi_value_mask_to_one_hot(self, mask: torch.Tensor | None):
        if mask is not None and self.has_mask:
            if self.multi_value_mask_max_classes is not None and mask.ndim == 3:
                # * if mask is dtype float, must be careful about its values, must be in [0, multi_value_mask_max_classes]
                # * if float value in it, the code will raise CUDA error by `scatter_` without clear hint.

                mask[mask >= self.multi_value_mask_max_classes] = (
                    0.0  # larger than max classes are set to background
                )
                new_mask = torch.zeros(
                    mask.size(0),
                    self.multi_value_mask_max_classes,
                    *mask.shape[-2:],
                    device=mask.device,
                    dtype=torch.float32,
                )
                new_mask.scatter_(1, mask.long()[:, None], 1.0)
            elif mask.ndim == 4:
                new_mask = mask
            else:
                raise ValueError(
                    f"Invalid mask shape: {mask.shape} when `multi_value_mask_max_classes` is {self.multi_value_mask_max_classes}",
                    "or model has `has_mask` as True. Input mask should be 3D or 4D tensor",
                    "such as [Bs, 1, H, W] or [Bs, *, H, W] (to be one-hot encoded in this case).",
                )
        else:
            new_mask = None

        return new_mask

    def fusion_train_step(
        self,
        vi: "torch.Tensor",
        ir: "torch.Tensor",
        mask: "torch.Tensor | None" = None,
        gt: "torch.Tensor | None" = None,
        txt: "torch.Tensor | None" = None,
        fusion_criterion: "Callable | None" = None,
        to_rgb_fn: "Callable | None" = None,
        has_gt: bool = False,
        **_kwargs,
    ):
        vi, ir, mask, txt = self.check_multi_modal_inputs(
            vi, ir, mask, txt, vi.device, vi.dtype
        )
        txt = self.drop_txt(txt)

        mask = self.check_multi_value_mask_to_one_hot(mask)

        fused_outp = self.only_fusion_step(vi, ir, mask, txt)

        # mask for loss, should be detached
        if mask is not None:
            mask_for_loss = mask.clone().detach()
        else:
            mask_for_loss = None

        fused_for_loss = to_rgb_fn(fused_outp) if to_rgb_fn is not None else fused_outp
        if (
            has_gt or gt.size(1) == 3
        ):  # TODO: find more robust way to check if gt is available
            # if we have supervised GT, we use it to compute the supervised loss
            # sometimes, for MEF fusion task, we can access the GTs
            fusion_gt = gt
            # two different modalities for GT to compute the unsupervised loss
            boundary_gt = (vi, ir)
        else:
            fusion_gt = None
            boundary_gt = (vi, ir)

        # compute supervised and unsupervised losses
        assert fusion_criterion is not None, "fusion_criterion should be provided"
        loss = list(
            fusion_criterion(
                fused_for_loss,
                boundary_gt=boundary_gt,
                fusion_gt=fusion_gt,
                mask=mask_for_loss,
            )
        )

        return fused_for_loss.clip(0, 1), loss

    @torch.no_grad()
    def fusion_val_step(
        self,
        vi: "torch.Tensor",
        ir: "torch.Tensor",
        mask: "torch.Tensor | None" = None,
        txt: "torch.Tensor | None" = None,
        *,
        patch_merge: bool = False,
        ret_seg_map: bool = False,
        to_rgb_fn: callable = None,
        **_kwargs,
    ):
        vi, ir, mask, txt = self.check_multi_modal_inputs(
            vi, ir, mask, txt, vi.device, vi.dtype
        )
        mask = self.check_multi_value_mask_to_one_hot(mask)

        fused_outp = self.only_fusion_step(vi, ir, mask, txt)

        # to rgb using SwinFusion CbCr fusion strategy
        if to_rgb_fn is not None:
            fused = to_rgb_fn(fused_outp)
        else:
            fused = fused_outp

        return fused.clip(0, 1)

    def patch_merge_step(self, lms, pan, **kwargs):
        if self.has_txt and "llm_feature" in kwargs:
            llm_feature = kwargs["llm_feature"]
            if llm_feature is not None:
                assert (
                    llm_feature.size(0) == 1
                ), "when using PatchMergeModule, input batch size should be 1"
                llm_feature = llm_feature.repeat(lms.size(0), 1, 1)
            kwargs["llm_feature"] = llm_feature

        fused = self._forward_implem(lms, pan, **kwargs)
        return fused

    def flops(self, inputs: tuple[torch.Tensor, ...]):
        img, cond, mask, llm_feature = inputs

        from fvcore.nn import FlopCountAnalysis, flop_count_table

        from model.module.rwkv_v5_multi_modal import RUN_CUDA_RWKV5_2, vrwkv6_flops

        self.forward = self._forward_implem
        flops_count = FlopCountAnalysis(self, (img, cond, mask, llm_feature))

        custom_ops = {"prim::PythonOp.WKV_5_2": vrwkv6_flops}
        flops_count.set_op_handle(**custom_ops)

        logger.info(flop_count_table(flops_count))


# * --- Test --- #


class RWKVFusionTester:
    def __init__(
        self,
        bs,
        img_size,
        task,
        model_size,
        with_mask=True,
        with_llm_feature=True,
        device="cuda:0",
        dtype=torch.float32,
        pt_path: str = None,
    ):
        # cfg 1 for pansharpening and HMIF
        cfg_pan_small = dict(
            img_channel=8,
            modal_channel=1,
            mask_channel=1 if with_mask else None,
            llm_channel=512 if with_llm_feature else None,
            out_channel=8,
            width=32,
            middle_blk_num=1,
            enc_blk_nums=[2, 1, 1],
            dec_blk_nums=[2, 1, 1],
            chan_upscales=[2, 1, 1],
            mid_window_size=0,
            enc_window_sizes=[0, 0, 0],
            dec_window_sizes=[0, 0, 0],
            MIFM_img_downsample_ratios=[4, 2, 1],
            ffn_hidden_rates=[2, 2, 2],
            with_checkpoint=False,
            pt_img_size=64,
            drop_path_rate=0.0,
            if_abs_pos=False,
            upscale=4,
            add_multi_modal_tokens=False,
            end_proj_type="3conv",
            fusion_prior="lr",
            scan_mode="K2",
            feature_prior=False,
            reconstruction_head="sr",
        )

        # cfg 2 for VIF
        cfg_vif_tiny = dict(
            img_channel=3,
            modal_channel=1,
            mask_channel=1 if with_mask else None,
            llm_channel=1024,
            out_channel=3,
            width=32,
            middle_blk_num=1,
            enc_blk_nums=[1, 1],
            dec_blk_nums=[1, 1],
            chan_upscales=[1, 1],
            mid_window_size=0,
            enc_window_sizes=[0, 0],
            dec_window_sizes=[0, 0],
            ffn_hidden_rates=[2, 2],
            MIFM_img_downsample_ratios=[1, 1],
            pt_img_size=[224, 280],
            drop_path_rate=0.2,
            drop_txt_ratio=0.0,
            txt_embd_drop_ratio=0.0,
            if_abs_pos=False,
            upscale=1,
            with_checkpoint=True,
            fusion_prior="none",
            add_multi_modal_tokens=False,
            end_proj_type="3conv",
            reconstruction_head="fusion",
            scan_mode="K2",
            shift_type="none",
        )
        # measure speed: 68 imgs/s
        cfg_vif_small = dict(
            img_channel=1,
            modal_channel=1,
            mask_channel=1 if with_mask else None,
            llm_channel=1024 if with_llm_feature else None,
            out_channel=1,
            width=32,
            middle_blk_num=1,
            enc_blk_nums=[1, 1],
            dec_blk_nums=[1, 1],
            chan_upscales=[1, 1],
            mid_window_size=0,
            enc_window_sizes=[0, 0],
            dec_window_sizes=[0, 0],
            ffn_hidden_rates=[2, 2],
            MIFM_img_downsample_ratios=[1, 1],
            with_checkpoint=True,
            pt_img_size=[224, 280],
            drop_path_rate=0.1,
            fusion_prior="none",
            add_multi_modal_tokens=False,
            end_proj_type="3conv",
            feature_prior=False,
            reconstruction_head="fusion",
            upscale=1,
        )
        cfg_mef_tiny = dict(
            img_channel=1,
            modal_channel=1,
            mask_channel=1 if with_mask else None,
            llm_channel=512 if with_llm_feature else None,
            multi_value_mask_max_classes=20 if with_mask else None,
            out_channel=1,
            width=32,
            middle_blk_num=1,
            enc_blk_nums=[1, 1, 1],
            dec_blk_nums=[1, 1, 1],
            chan_upscales=[2, 1, 1],
            mid_window_size=0,
            enc_window_sizes=[0, 0, 0],
            dec_window_sizes=[0, 0, 0],
            ffn_hidden_rates=[4, 2, 2],
            MIFM_img_downsample_ratios=[8, 8, 4],
            with_checkpoint=True,
            pt_img_size=[256, 256],
            drop_path_rate=0.0,
            fusion_prior="none",
            add_multi_modal_tokens=False,
            end_proj_type="3conv",
            feature_prior=False,
            reconstruction_head="fusion",
            upscale=1,
            scan_mode="K2",
            shift_type="none",
            add_block_residual=False,
        )
        cfgs = {
            "pan_small": cfg_pan_small,
            "vif_tiny": cfg_vif_tiny,
            "vif_small": cfg_vif_small,
            "mef_tiny": cfg_mef_tiny,
        }
        logger.info(f"using dtype: {dtype}")

        self.cfg = cfgs[task + "_" + model_size]
        self.bs = bs
        self.dtype = dtype
        self.img_size = (
            img_size if isinstance(img_size, tuple) else (img_size, img_size)
        )
        self.task = task
        self.device = device
        self.with_mask = with_mask
        self.with_llm_feature = with_llm_feature

        self.scale = self.cfg["upscale"]
        self.channel_img = {"pan": 8, "vif": 1, "mef": 1}[task]
        self.channel_cond = {"pan": 1, "vif": 1, "mef": 1}[task]

        self._logger = easy_logger(func_name="model_test")
        self.print = self._logger.info
        self.info = self._logger.info
        self.debug = self._logger.debug
        self.warning = self._logger.warning
        self.error = self._logger.error

        ## init model
        self.net = RWKVFusion(**self.cfg).to(device)

        ## load checkpoint
        if pt_path is not None:
            from utils import module_load

            self.net = module_load(pt_path, self.net, device, strict=True)

    def prepare_data(self, real_img=False):
        import kornia
        from torchvision.io import ImageReadMode, read_image

        def two_color_fusion(modal_1, modal_2):
            _s1_y_cb_cr = kornia.color.rgb_to_ycbcr(modal_1)
            if modal_2.shape[1] == 1:
                modal_2_r = modal_2.repeat(1, 3, 1, 1)
            else:
                modal_2_r = modal_2
            _s2_y_cb_cr = kornia.color.rgb_to_ycbcr(modal_2_r)

            cbcr_1 = _s1_y_cb_cr[:, 1:]
            cbcr_2 = _s2_y_cb_cr[:, 1:]

            def back_to_rgb(pred_y):
                tau = 0.5
                mid_1 = cbcr_1 * torch.abs(cbcr_1 - tau) + cbcr_2 * torch.abs(
                    cbcr_2 - tau
                )
                mid_2 = torch.abs(cbcr_1 - tau) + torch.abs(cbcr_2 - tau)
                _mask = mid_2 == 0

                cbcr_fused = mid_1 / mid_2
                cbcr_fused[_mask] = tau

                y_cb_cr = torch.cat([pred_y, cbcr_fused], dim=1)
                return kornia.color.ycbcr_to_rgb(y_cb_cr).clip(0, 1)

            return back_to_rgb

        ## test forward and backward
        # for pan task: img is lms, cond is pan
        # for vif task: img is vis, cond is ir
        to_rgb_fn = None
        if real_img:
            img = (
                read_image("/Data4/cao/ZiHanCao/datasets/VIF-MSRS/train/vi/00891N.jpg")[
                    None, ...
                ].to(device)
                / 255.0
            )
            cond = (
                read_image("/Data4/cao/ZiHanCao/datasets/VIF-MSRS/train/ir/00891N.jpg")[
                    None, ...
                ].to(device)
                / 255.0
            )
            # resize
            img = F.interpolate(img, self.img_size, mode="nearest", align_corners=None)
            cond = F.interpolate(
                cond, self.img_size, mode="nearest", align_corners=None
            )
            # rgb fn
            to_rgb_fn = two_color_fusion(img, cond)
            # to gray
            img = kornia.color.rgb_to_grayscale(img)
            if cond.shape[1] == 3:
                cond = kornia.color.rgb_to_grayscale(cond)
        else:
            img = torch.randn(
                self.bs,
                self.channel_img,
                self.img_size[0],
                self.img_size[1],
                dtype=self.dtype,
            ).to(device)
            cond = torch.randn(
                self.bs,
                self.channel_cond,
                self.img_size[0],
                self.img_size[1],
                dtype=self.dtype,
            ).to(device)

        if self.with_mask:
            if real_img:
                mask = read_image(
                    "/Data4/cao/ZiHanCao/datasets/VIF-MSRS/train/Segmentation_labels/00891N.png"
                ).to(device)
                mask = F.interpolate(mask[:, None], self.img_size, mode="nearest")[:, 0]
                # mask = self.net.check_multi_value_mask_to_one_hot(mask)
                mask = mask[:, None].float()
                mask[mask > 0.0] = 1.0
            else:
                if self.net.multi_value_mask_max_classes is not None:
                    mask = torch.ones(
                        (
                            self.bs,
                            self.net.multi_value_mask_max_classes,
                            self.img_size[0],
                            self.img_size[1],
                        ),
                        dtype=self.dtype,
                    ).to(img)
                else:
                    mask = torch.randint(
                        0,
                        9,
                        (self.bs, 1, self.img_size[0], self.img_size[1]),
                        dtype=self.dtype,
                    ).to(img)
        else:
            mask = None

        if self.with_llm_feature:
            if real_img:
                from safetensors import safe_open

                llm_file_vi = safe_open(
                    "/Data4/cao/ZiHanCao/datasets/VIF-MSRS/train/t5_feature_MSRS_train_vi.safetensors",
                    framework="pt",
                )
                llm_file_ir = safe_open(
                    "/Data4/cao/ZiHanCao/datasets/VIF-MSRS/train/t5_feature_MSRS_train_ir.safetensors",
                    framework="pt",
                )
                llm_feature_vi = llm_file_vi.get_tensor("00891N").to(
                    device, dtype=self.dtype
                )
                llm_feature_ir = llm_file_ir.get_tensor("00891N").to(
                    device, dtype=self.dtype
                )
                llm_feature = torch.cat([llm_feature_vi, llm_feature_ir], dim=-1)
            else:
                llm_feature = torch.randn(
                    self.bs, 512, self.cfg["llm_channel"], dtype=self.dtype
                ).to(device)
        else:
            llm_feature = None

        return img, cond, mask, llm_feature, to_rgb_fn

    def summary_mem(self):
        ## memory usage
        self.print("Memory usage:")
        self.print(memory_summary(device=device, abbreviated=True))

    def params_and_flops(self, params_only=True):
        ## Count params, flops
        from fvcore.nn import (
            FlopCountAnalysis,
            flop_count_table,
            parameter_count_table,
        )

        from model.module.rwkv_v2 import vrwkv6_flops
        from model.module.rwkv_v5_multi_modal import WKV_5_2

        self.net.forward = self.net._forward_implem
        if not params_only:
            img, cond, mask, llm_feature, _ = self.prepare_data()
            flops_count = FlopCountAnalysis(self.net, (img, cond, mask, llm_feature))

            custom_ops = {"prim::PythonOp.WKV_5_2": vrwkv6_flops}
            flops_count.set_op_handle(**custom_ops)

            self.print(flop_count_table(flops_count))
        else:
            from fvcore.nn import parameter_count_table

            self.print(parameter_count_table(self.net))

    @torch.no_grad()
    def measure_throughput(self, batch_size: int = 56):
        # assert not self.with_mask, '`with_mask` is not allowed in throughput measurement'
        self.info(f"measure throughput for task: {self.task}")

        ## measure throughput
        from utils import measure_throughput

        self.net.forward = self.net._forward_implem
        throghtput = measure_throughput(
            self.net,
            [
                (self.channel_img, self.img_size[0], self.img_size[1]),
                (self.channel_cond, self.img_size[0], self.img_size[1]),
                (1, self.img_size[0], self.img_size[1]),
                (512, self.cfg["llm_channel"]),
            ],
            batch_size,
            num_warmup=10,
            num_iterations=50,
        )

        self.print(f"throughput: {throghtput} imgs/s")

    # @torch.no_grad()
    def test_forward(self):
        img, cond, mask, llm_feature, _ = self.prepare_data()
        with torch.autocast(device_type="cuda", dtype=self.dtype):
            out = self.net._forward_implem(img, cond, mask, llm_feature)
        self.info(f"output shape: {out.shape} with dtype: {out.dtype}")
        # self.summary_mem()

    def test_forward_backward(self, real_img=False):
        img, cond, mask, llm_feature, _ = self.prepare_data(real_img=real_img)
        out = self.net._forward_implem(img, cond, mask, llm_feature)
        sr = torch.randn(
            self.bs, self.channel_img, self.img_size[0], self.img_size[1]
        ).to(device)
        logger.info(f"output shape: {out.shape} with dtype: {out.dtype}")
        loss = F.mse_loss(out, sr)
        logger.print(loss)
        loss.backward()

        # find unused params and big-normed gradient
        d_grads = {}
        n_params = 0
        for n, p in self.net.named_parameters():
            n_params += p.numel()
            if p.grad is None:
                print(n, "has no grad")
            else:
                p_sum = torch.abs(p.grad).sum().item()
                d_grads[n] = p_sum

        ## topk
        d_grads = dict(sorted(d_grads.items(), key=lambda item: item[1], reverse=True))
        for k, v in list(d_grads.items())[:20]:
            self.print(k, v)

        ## params
        self.print("total params:", n_params / 1e6, "M")
        from fvcore.nn import parameter_count_table

        self.print(parameter_count_table(self.net))

        ## memory usage
        self.summary_mem()

    @catch_any_error
    def test_val_step(self):
        if self.task == "pan":
            img, cond, mask, llm_feature, _ = self.prepare_data()
            fused = self.net.sharpening_val_step(
                img, cond, llm_feature, patch_merge=True
            )
            self.info(f"fused shape: {fused.shape}")
            self.summary_mem()
        elif self.task == "vif":
            # self.warning('use patch_merge_step only for sharpening task')
            img, cond, mask, llm_feature, _ = self.prepare_data()
            fused = self.net.fusion_val_step(img, cond, mask, llm_feature)
            self.info(f"fused shape: {fused.shape}")
            self.summary_mem()
        elif self.task == "mef":
            img, cond, mask, llm_feature, to_rgb_fn = self.prepare_data(True)
            fused = self.net.fusion_val_step(
                img, cond, mask, llm_feature, to_rgb_fn=to_rgb_fn
            )
            self.info(f"fused shape: {fused.shape}")
            self.summary_mem()

    def test_params(
        self,
    ):
        for n, p in self.net.named_parameters():
            if p.dtype != torch.float32:
                logger.warning(f"{n} has dtype: {p.dtype}")
            if not p.is_floating_point():
                logger.warning(f"{n} is not floating point, has dtype: {p.dtype}")

        logger.info(f"total params: {sum(p.numel() for p in self.net.parameters())}")

    def test_fused_optimizer(
        self,
    ):
        import torch.optim as optim

        from utils.optim_utils import get_optimizer

        optimizer = optim.AdamW(
            self.net.parameters(), lr=1e-3, fused=False, foreach=True
        )
        # optimizer = get_optimizer(self.net, self.net.parameters(), lr=1e-3,
        #                           name='shampoo_ddp', use_pytorch_compile=False)
        img, cond, mask, llm_feature, _ = self.prepare_data()
        sr = torch.randn(
            self.bs, self.channel_img, self.img_size[0], self.img_size[1]
        ).to(device)
        out = self.net._forward_implem(img, cond, mask, llm_feature)
        loss = F.mse_loss(out, sr)
        loss.backward()
        optimizer.step()

        logger.info(f"optimizer: {optimizer} step done.")


if __name__ == "__main__":
    from torch.cuda import memory_summary

    from utils import catch_any_error

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    # img size: (280, 224) for VIF, and (64, 64) for PAN - training
    # img size: (672, 504) for VIF (padder with base 56), and (256, 256) for PAN - testing
    # (768, 1024)

    tester = RWKVFusionTester(
        1,
        (224, 280),
        "vif",
        "small",
        with_mask=True,
        with_llm_feature=True,
        device=device,
        dtype=torch.float32,
        # pt_path="/Data4/cao/ZiHanCao/exps/panformer/log_file/RWKVFusion_v12_RWKVFusion/vis_ir_joint/2024-11-08-14-44-12_panRWKV_sb45j3w5_vis_ir_joint_rwkv5_2_wo_omnishift_lerp_factor=0/weights/ema_model.pth/model.safetensors",
    )

    # tester.params_and_flops(False)
    # tester.test_forward()
    tester.test_forward_backward(True)
    # tester.test_val_step()
    # tester.measure_throughput()
    # tester.test_params()
    # tester.test_fused_optimizer()
