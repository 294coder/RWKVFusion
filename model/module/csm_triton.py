# triton cross scan, 2x speed than pytorch implementation =========================
import torch
import triton
import triton.language as tl


@triton.jit
def triton_cross_scan(
    x,  # (B, C, H, W)
    y,  # (B, 4, C, H, W)
    BC: tl.constexpr,
    BH: tl.constexpr,
    BW: tl.constexpr,
    DC: tl.constexpr,
    DH: tl.constexpr,
    DW: tl.constexpr,
    NH: tl.constexpr,
    NW: tl.constexpr,
):
    i_hw, i_c, i_b = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_h, i_w = (i_hw // NW), (i_hw % NW)
    _mask_h = (i_h * BH + tl.arange(0, BH)) < DH
    _mask_w = (i_w * BW + tl.arange(0, BW)) < DW
    _mask_hw = _mask_h[:, None] & _mask_w[None, :]
    _for_C = min(DC - i_c * BC, BC)

    _tmp0 = i_c * BC * DH * DW
    _tmp1 = DC * DH * DW
    _tmp2 = (
        _tmp0
        + i_h * BH * DW
        + tl.arange(0, BH)[:, None] * DW
        + i_w * BW
        + tl.arange(0, BW)[None, :]
    )
    p_x = x + i_b * _tmp1 + _tmp2
    p_y1 = y + i_b * 4 * _tmp1 + _tmp2  # same
    p_y2 = (
        y
        + i_b * 4 * _tmp1
        + _tmp1
        + _tmp0
        + i_w * BW * DH
        + tl.arange(0, BW)[None, :] * DH
        + i_h * BH
        + tl.arange(0, BH)[:, None]
    )  # trans
    p_y3 = (
        y
        + i_b * 4 * _tmp1
        + 2 * _tmp1
        + _tmp0
        + (NH - i_h - 1) * BH * DW
        + (BH - 1 - tl.arange(0, BH)[:, None]) * DW
        + (NW - i_w - 1) * BW
        + (BW - 1 - tl.arange(0, BW)[None, :])
        + (DH - NH * BH) * DW
        + (DW - NW * BW)
    )  # flip
    p_y4 = (
        y
        + i_b * 4 * _tmp1
        + 3 * _tmp1
        + _tmp0
        + (NW - i_w - 1) * BW * DH
        + (BW - 1 - tl.arange(0, BW)[None, :]) * DH
        + (NH - i_h - 1) * BH
        + (BH - 1 - tl.arange(0, BH)[:, None])
        + (DH - NH * BH)
        + (DW - NW * BW) * DH
    )  # trans + flip

    for idxc in range(_for_C):
        _idx = idxc * DH * DW
        _x = tl.load(p_x + _idx, mask=_mask_hw)
        tl.store(p_y1 + _idx, _x, mask=_mask_hw)
        tl.store(p_y2 + _idx, _x, mask=_mask_hw)
        tl.store(p_y3 + _idx, _x, mask=_mask_hw)
        tl.store(p_y4 + _idx, _x, mask=_mask_hw)
    tl.debug_barrier()


@triton.jit
def triton_cross_merge(
    x,  # (B, C, H, W)
    y,  # (B, 4, C, H, W)
    BC: tl.constexpr,
    BH: tl.constexpr,
    BW: tl.constexpr,
    DC: tl.constexpr,
    DH: tl.constexpr,
    DW: tl.constexpr,
    NH: tl.constexpr,
    NW: tl.constexpr,
):
    i_hw, i_c, i_b = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_h, i_w = (i_hw // NW), (i_hw % NW)
    _mask_h = (i_h * BH + tl.arange(0, BH)) < DH
    _mask_w = (i_w * BW + tl.arange(0, BW)) < DW
    _mask_hw = _mask_h[:, None] & _mask_w[None, :]
    _for_C = min(DC - i_c * BC, BC)

    _tmp0 = i_c * BC * DH * DW
    _tmp1 = DC * DH * DW
    _tmp2 = (
        _tmp0
        + i_h * BH * DW
        + tl.arange(0, BH)[:, None] * DW
        + i_w * BW
        + tl.arange(0, BW)[None, :]
    )
    p_x = x + i_b * _tmp1 + _tmp2
    p_y1 = y + i_b * 4 * _tmp1 + _tmp2  # same
    p_y2 = (
        y
        + i_b * 4 * _tmp1
        + _tmp1
        + _tmp0
        + i_w * BW * DH
        + tl.arange(0, BW)[None, :] * DH
        + i_h * BH
        + tl.arange(0, BH)[:, None]
    )  # trans
    p_y3 = (
        y
        + i_b * 4 * _tmp1
        + 2 * _tmp1
        + _tmp0
        + (NH - i_h - 1) * BH * DW
        + (BH - 1 - tl.arange(0, BH)[:, None]) * DW
        + (NW - i_w - 1) * BW
        + (BW - 1 - tl.arange(0, BW)[None, :])
        + (DH - NH * BH) * DW
        + (DW - NW * BW)
    )  # flip
    p_y4 = (
        y
        + i_b * 4 * _tmp1
        + 3 * _tmp1
        + _tmp0
        + (NW - i_w - 1) * BW * DH
        + (BW - 1 - tl.arange(0, BW)[None, :]) * DH
        + (NH - i_h - 1) * BH
        + (BH - 1 - tl.arange(0, BH)[:, None])
        + (DH - NH * BH)
        + (DW - NW * BW) * DH
    )  # trans + flip

    for idxc in range(_for_C):
        _idx = idxc * DH * DW
        _y1 = tl.load(p_y1 + _idx, mask=_mask_hw)
        _y2 = tl.load(p_y2 + _idx, mask=_mask_hw)
        _y3 = tl.load(p_y3 + _idx, mask=_mask_hw)
        _y4 = tl.load(p_y4 + _idx, mask=_mask_hw)
        tl.store(p_x + _idx, _y1 + _y2 + _y3 + _y4, mask=_mask_hw)
    tl.debug_barrier()


@triton.jit
def triton_cross_scan_1b1(
    x,  # (B, C, H, W)
    y,  # (B, 4, C, H, W)
    BC: tl.constexpr,
    BH: tl.constexpr,
    BW: tl.constexpr,
    DC: tl.constexpr,
    DH: tl.constexpr,
    DW: tl.constexpr,
    NH: tl.constexpr,
    NW: tl.constexpr,
):
    i_hw, i_c, i_b = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_h, i_w = (i_hw // NW), (i_hw % NW)
    _mask_h = (i_h * BH + tl.arange(0, BH)) < DH
    _mask_w = (i_w * BW + tl.arange(0, BW)) < DW
    _mask_hw = _mask_h[:, None] & _mask_w[None, :]
    _for_C = min(DC - i_c * BC, BC)

    _tmp0 = i_c * BC * DH * DW
    _tmp1 = DC * DH * DW
    _tmp2 = (
        _tmp0
        + i_h * BH * DW
        + tl.arange(0, BH)[:, None] * DW
        + i_w * BW
        + tl.arange(0, BW)[None, :]
    )
    p_y1 = y + i_b * 4 * _tmp1 + _tmp2  # same
    p_y2 = (
        y
        + i_b * 4 * _tmp1
        + _tmp1
        + _tmp0
        + i_w * BW * DH
        + tl.arange(0, BW)[None, :] * DH
        + i_h * BH
        + tl.arange(0, BH)[:, None]
    )  # trans
    p_y3 = (
        y
        + i_b * 4 * _tmp1
        + 2 * _tmp1
        + _tmp0
        + (NH - i_h - 1) * BH * DW
        + (BH - 1 - tl.arange(0, BH)[:, None]) * DW
        + (NW - i_w - 1) * BW
        + (BW - 1 - tl.arange(0, BW)[None, :])
        + (DH - NH * BH) * DW
        + (DW - NW * BW)
    )  # flip
    p_y4 = (
        y
        + i_b * 4 * _tmp1
        + 3 * _tmp1
        + _tmp0
        + (NW - i_w - 1) * BW * DH
        + (BW - 1 - tl.arange(0, BW)[None, :]) * DH
        + (NH - i_h - 1) * BH
        + (BH - 1 - tl.arange(0, BH)[:, None])
        + (DH - NH * BH)
        + (DW - NW * BW) * DH
    )  # trans + flip

    p_x1 = x + i_b * 4 * _tmp1 + _tmp2
    p_x2 = p_x1 + _tmp1
    p_x3 = p_x2 + _tmp1
    p_x4 = p_x3 + _tmp1
    for idxc in range(_for_C):
        _idx = idxc * DH * DW
        tl.store(p_y1 + _idx, tl.load(p_x1 + _idx), mask=_mask_hw)
        tl.store(p_y2 + _idx, tl.load(p_x2 + _idx), mask=_mask_hw)
        tl.store(p_y3 + _idx, tl.load(p_x3 + _idx), mask=_mask_hw)
        tl.store(p_y4 + _idx, tl.load(p_x4 + _idx), mask=_mask_hw)
    tl.debug_barrier()


@triton.jit
def triton_cross_merge_1b1(
    x,  # (B, C, H, W)
    y,  # (B, 4, C, H, W)
    BC: tl.constexpr,
    BH: tl.constexpr,
    BW: tl.constexpr,
    DC: tl.constexpr,
    DH: tl.constexpr,
    DW: tl.constexpr,
    NH: tl.constexpr,
    NW: tl.constexpr,
):
    i_hw, i_c, i_b = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_h, i_w = (i_hw // NW), (i_hw % NW)
    _mask_h = (i_h * BH + tl.arange(0, BH)) < DH
    _mask_w = (i_w * BW + tl.arange(0, BW)) < DW
    _mask_hw = _mask_h[:, None] & _mask_w[None, :]
    _for_C = min(DC - i_c * BC, BC)

    _tmp0 = i_c * BC * DH * DW
    _tmp1 = DC * DH * DW
    _tmp2 = (
        _tmp0
        + i_h * BH * DW
        + tl.arange(0, BH)[:, None] * DW
        + i_w * BW
        + tl.arange(0, BW)[None, :]
    )
    p_y1 = y + i_b * 4 * _tmp1 + _tmp2  # same
    p_y2 = (
        y
        + i_b * 4 * _tmp1
        + _tmp1
        + _tmp0
        + i_w * BW * DH
        + tl.arange(0, BW)[None, :] * DH
        + i_h * BH
        + tl.arange(0, BH)[:, None]
    )  # trans
    p_y3 = (
        y
        + i_b * 4 * _tmp1
        + 2 * _tmp1
        + _tmp0
        + (NH - i_h - 1) * BH * DW
        + (BH - 1 - tl.arange(0, BH)[:, None]) * DW
        + (NW - i_w - 1) * BW
        + (BW - 1 - tl.arange(0, BW)[None, :])
        + (DH - NH * BH) * DW
        + (DW - NW * BW)
    )  # flip
    p_y4 = (
        y
        + i_b * 4 * _tmp1
        + 3 * _tmp1
        + _tmp0
        + (NW - i_w - 1) * BW * DH
        + (BW - 1 - tl.arange(0, BW)[None, :]) * DH
        + (NH - i_h - 1) * BH
        + (BH - 1 - tl.arange(0, BH)[:, None])
        + (DH - NH * BH)
        + (DW - NW * BW) * DH
    )  # trans + flip

    p_x1 = x + i_b * 4 * _tmp1 + _tmp2
    p_x2 = p_x1 + _tmp1
    p_x3 = p_x2 + _tmp1
    p_x4 = p_x3 + _tmp1
    for idxc in range(_for_C):
        _idx = idxc * DH * DW
        tl.store(p_x1 + _idx, tl.load(p_y1 + _idx), mask=_mask_hw)
        tl.store(p_x2 + _idx, tl.load(p_y2 + _idx), mask=_mask_hw)
        tl.store(p_x3 + _idx, tl.load(p_y3 + _idx), mask=_mask_hw)
        tl.store(p_x4 + _idx, tl.load(p_y4 + _idx), mask=_mask_hw)
    tl.debug_barrier()


### same and trans
@triton.jit
def triton_cross_scan_same_and_trans(
    x,  # (B, C, H, W)
    y,  # (B, 2, C, H, W)
    BC: tl.constexpr,
    BH: tl.constexpr,
    BW: tl.constexpr,
    DC: tl.constexpr,
    DH: tl.constexpr,
    DW: tl.constexpr,
    NH: tl.constexpr,
    NW: tl.constexpr,
):
    i_hw, i_c, i_b = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_h, i_w = (i_hw // NW), (i_hw % NW)
    _mask_h = (i_h * BH + tl.arange(0, BH)) < DH
    _mask_w = (i_w * BW + tl.arange(0, BW)) < DW
    _mask_hw = _mask_h[:, None] & _mask_w[None, :]
    _for_C = min(DC - i_c * BC, BC)

    _tmp0 = i_c * BC * DH * DW
    _tmp1 = DC * DH * DW
    _tmp2 = (
        _tmp0
        + i_h * BH * DW
        + tl.arange(0, BH)[:, None] * DW
        + i_w * BW
        + tl.arange(0, BW)[None, :]
    )
    p_x = x + i_b * _tmp1 + _tmp2
    p_y1 = y + i_b * 2 * _tmp1 + _tmp2  # same - 注意这里的索引从4变为2
    p_y2 = (
        y
        + i_b * 2 * _tmp1
        + _tmp1
        + _tmp0
        + i_w * BW * DH
        + tl.arange(0, BW)[None, :] * DH
        + i_h * BH
        + tl.arange(0, BH)[:, None]
    )  # trans

    for idxc in range(_for_C):
        _idx = idxc * DH * DW
        _x = tl.load(p_x + _idx, mask=_mask_hw)
        tl.store(p_y1 + _idx, _x, mask=_mask_hw)
        tl.store(p_y2 + _idx, _x, mask=_mask_hw)
    tl.debug_barrier()


@triton.jit
def triton_cross_merge_same_and_trans(
    x,  # (B, C, H, W)
    y,  # (B, 2, C, H, W)
    BC: tl.constexpr,
    BH: tl.constexpr,
    BW: tl.constexpr,
    DC: tl.constexpr,
    DH: tl.constexpr,
    DW: tl.constexpr,
    NH: tl.constexpr,
    NW: tl.constexpr,
):
    i_hw, i_c, i_b = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_h, i_w = (i_hw // NW), (i_hw % NW)
    _mask_h = (i_h * BH + tl.arange(0, BH)) < DH
    _mask_w = (i_w * BW + tl.arange(0, BW)) < DW
    _mask_hw = _mask_h[:, None] & _mask_w[None, :]
    _for_C = min(DC - i_c * BC, BC)

    _tmp0 = i_c * BC * DH * DW
    _tmp1 = DC * DH * DW
    _tmp2 = (
        _tmp0
        + i_h * BH * DW
        + tl.arange(0, BH)[:, None] * DW
        + i_w * BW
        + tl.arange(0, BW)[None, :]
    )
    p_x = x + i_b * _tmp1 + _tmp2
    p_y1 = y + i_b * 2 * _tmp1 + _tmp2  # same
    p_y2 = (
        y
        + i_b * 2 * _tmp1
        + _tmp1
        + _tmp0
        + i_w * BW * DH
        + tl.arange(0, BW)[None, :] * DH
        + i_h * BH
        + tl.arange(0, BH)[:, None]
    )  # trans

    for idxc in range(_for_C):
        _idx = idxc * DH * DW
        _y1 = tl.load(p_y1 + _idx, mask=_mask_hw)
        _y2 = tl.load(p_y2 + _idx, mask=_mask_hw)
        tl.store(p_x + _idx, _y1 + _y2, mask=_mask_hw)
    tl.debug_barrier()


### flip and trans-with-flip
@triton.jit
def triton_cross_scan_trans_and_flips(
    x,  # (B, C, H, W)
    y,  # (B, 2, C, H, W)
    BC: tl.constexpr,
    BH: tl.constexpr,
    BW: tl.constexpr,
    DC: tl.constexpr,
    DH: tl.constexpr,
    DW: tl.constexpr,
    NH: tl.constexpr,
    NW: tl.constexpr,
):
    i_hw, i_c, i_b = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_h, i_w = (i_hw // NW), (i_hw % NW)
    _mask_h = (i_h * BH + tl.arange(0, BH)) < DH
    _mask_w = (i_w * BW + tl.arange(0, BW)) < DW
    _mask_hw = _mask_h[:, None] & _mask_w[None, :]
    _for_C = min(DC - i_c * BC, BC)

    _tmp0 = i_c * BC * DH * DW
    _tmp1 = DC * DH * DW
    _tmp2 = (
        _tmp0
        + i_h * BH * DW
        + tl.arange(0, BH)[:, None] * DW
        + i_w * BW
        + tl.arange(0, BW)[None, :]
    )
    p_x = x + i_b * _tmp1 + _tmp2

    p_y3 = (
        y
        + i_b * 2 * _tmp1
        + _tmp0
        + (NH - i_h - 1) * BH * DW
        + (BH - 1 - tl.arange(0, BH)[:, None]) * DW
        + (NW - i_w - 1) * BW
        + (BW - 1 - tl.arange(0, BW)[None, :])
        + (DH - NH * BH) * DW
        + (DW - NW * BW)
    )  # flip
    p_y4 = (
        y
        + i_b * 2 * _tmp1
        + _tmp1
        + _tmp0
        + (NW - i_w - 1) * BW * DH
        + (BW - 1 - tl.arange(0, BW)[None, :]) * DH
        + (NH - i_h - 1) * BH
        + (BH - 1 - tl.arange(0, BH)[:, None])
        + (DH - NH * BH)
        + (DW - NW * BW) * DH
    )  # trans + flip

    for idxc in range(_for_C):
        _idx = idxc * DH * DW
        _x = tl.load(p_x + _idx, mask=_mask_hw)
        # tl.store(p_y1 + _idx, _x, mask=_mask_hw)
        # tl.store(p_y2 + _idx, _x, mask=_mask_hw)
        tl.store(p_y3 + _idx, _x, mask=_mask_hw)
        tl.store(p_y4 + _idx, _x, mask=_mask_hw)
    tl.debug_barrier()


@triton.jit
def triton_cross_merge_trans_and_flips(
    x,  # (B, C, H, W)
    y,  # (B, 2, C, H, W)
    BC: tl.constexpr,
    BH: tl.constexpr,
    BW: tl.constexpr,
    DC: tl.constexpr,
    DH: tl.constexpr,
    DW: tl.constexpr,
    NH: tl.constexpr,
    NW: tl.constexpr,
):
    i_hw, i_c, i_b = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_h, i_w = (i_hw // NW), (i_hw % NW)
    _mask_h = (i_h * BH + tl.arange(0, BH)) < DH
    _mask_w = (i_w * BW + tl.arange(0, BW)) < DW
    _mask_hw = _mask_h[:, None] & _mask_w[None, :]
    _for_C = min(DC - i_c * BC, BC)

    _tmp0 = i_c * BC * DH * DW
    _tmp1 = DC * DH * DW
    _tmp2 = (
        _tmp0
        + i_h * BH * DW
        + tl.arange(0, BH)[:, None] * DW
        + i_w * BW
        + tl.arange(0, BW)[None, :]
    )
    p_x = x + i_b * _tmp1 + _tmp2

    # p_y3 = y + i_b * 2 * _tmp1 + _tmp1 + _tmp0 + (NH - i_h - 1) * BH * DW  + (BH - 1 - tl.arange(0, BH)[:, None]) * DW + (NW - i_w - 1) * BW + (BW - 1 - tl.arange(0, BW)[None, :]) + (DH - NH * BH) * DW + (DW - NW * BW) # flip
    # p_y4 = y + i_b * 2 * _tmp1 + _tmp1 + _tmp0 + (NW - i_w - 1) * BW * DH  + (BW - 1 - tl.arange(0, BW)[None, :]) * DH + (NH - i_h - 1) * BH + (BH - 1 - tl.arange(0, BH)[:, None]) + (DH - NH * BH) + (DW - NW * BW) * DH  # trans + flip

    p_y3 = (
        y
        + i_b * 2 * _tmp1
        + _tmp0
        + (NH - i_h - 1) * BH * DW
        + (BH - 1 - tl.arange(0, BH)[:, None]) * DW
        + (NW - i_w - 1) * BW
        + (BW - 1 - tl.arange(0, BW)[None, :])
        + (DH - NH * BH) * DW
        + (DW - NW * BW)
    )  # flip
    p_y4 = (
        y
        + i_b * 2 * _tmp1
        + _tmp1
        + _tmp0
        + (NW - i_w - 1) * BW * DH
        + (BW - 1 - tl.arange(0, BW)[None, :]) * DH
        + (NH - i_h - 1) * BH
        + (BH - 1 - tl.arange(0, BH)[:, None])
        + (DH - NH * BH)
        + (DW - NW * BW) * DH
    )  # trans + flip

    for idxc in range(_for_C):
        _idx = idxc * DH * DW
        _y3 = tl.load(p_y3 + _idx, mask=_mask_hw)
        _y4 = tl.load(p_y4 + _idx, mask=_mask_hw)
        tl.store(p_x + _idx, _y3 + _y4, mask=_mask_hw)
    tl.debug_barrier()


##############################################################
@triton.jit
def triton_cross_scan_same(
    x,  # (B, C, H, W)
    y,  # (B, 1, C, H, W)
    BC: tl.constexpr,
    BH: tl.constexpr,
    BW: tl.constexpr,
    DC: tl.constexpr,
    DH: tl.constexpr,
    DW: tl.constexpr,
    NH: tl.constexpr,
    NW: tl.constexpr,
):
    i_hw, i_c, i_b = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_h, i_w = (i_hw // NW), (i_hw % NW)
    _mask_h = (i_h * BH + tl.arange(0, BH)) < DH
    _mask_w = (i_w * BW + tl.arange(0, BW)) < DW
    _mask_hw = _mask_h[:, None] & _mask_w[None, :]
    _for_C = min(DC - i_c * BC, BC)

    _tmp0 = i_c * BC * DH * DW
    _tmp1 = DC * DH * DW
    _tmp2 = (
        _tmp0
        + i_h * BH * DW
        + tl.arange(0, BH)[:, None] * DW
        + i_w * BW
        + tl.arange(0, BW)[None, :]
    )
    p_x = x + i_b * _tmp1 + _tmp2
    p_y1 = y + i_b * _tmp1 + _tmp2  # same - 注意这里的索引从4变为2
    # p_y2 = y + i_b * 2 * _tmp1 + _tmp1 + _tmp0 + i_w * BW * DH + tl.arange(0, BW)[None, :] * DH + i_h * BH + tl.arange(0, BH)[:, None]  # trans

    for idxc in range(_for_C):
        _idx = idxc * DH * DW
        _x = tl.load(p_x + _idx, mask=_mask_hw)
        tl.store(p_y1 + _idx, _x, mask=_mask_hw)
        # tl.store(p_y2 + _idx, _x, mask=_mask_hw)
    tl.debug_barrier()


@triton.jit
def triton_cross_merge_same(
    x,  # (B, C, H, W)
    y,  # (B, 1, C, H, W)
    BC: tl.constexpr,
    BH: tl.constexpr,
    BW: tl.constexpr,
    DC: tl.constexpr,
    DH: tl.constexpr,
    DW: tl.constexpr,
    NH: tl.constexpr,
    NW: tl.constexpr,
):
    i_hw, i_c, i_b = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_h, i_w = (i_hw // NW), (i_hw % NW)
    _mask_h = (i_h * BH + tl.arange(0, BH)) < DH
    _mask_w = (i_w * BW + tl.arange(0, BW)) < DW
    _mask_hw = _mask_h[:, None] & _mask_w[None, :]
    _for_C = min(DC - i_c * BC, BC)

    _tmp0 = i_c * BC * DH * DW
    _tmp1 = DC * DH * DW
    _tmp2 = (
        _tmp0
        + i_h * BH * DW
        + tl.arange(0, BH)[:, None] * DW
        + i_w * BW
        + tl.arange(0, BW)[None, :]
    )
    p_x = x + i_b * _tmp1 + _tmp2

    p_y = y + i_b * _tmp1 + _tmp2

    for idxc in range(_for_C):
        _idx = idxc * DH * DW
        _y1 = tl.load(p_y + _idx, mask=_mask_hw)
        # _y2 = tl.load(p_y2 + _idx, mask=_mask_hw)
        tl.store(p_x + _idx, _y1, mask=_mask_hw)
    tl.debug_barrier()


@triton.jit
def triton_cross_scan_trans(
    x,  # (B, C, H, W)
    y,  # (B, 1, C, H, W)
    BC: tl.constexpr,
    BH: tl.constexpr,
    BW: tl.constexpr,
    DC: tl.constexpr,
    DH: tl.constexpr,
    DW: tl.constexpr,
    NH: tl.constexpr,
    NW: tl.constexpr,
):
    i_hw, i_c, i_b = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_h, i_w = (i_hw // NW), (i_hw % NW)
    _mask_h = (i_h * BH + tl.arange(0, BH)) < DH
    _mask_w = (i_w * BW + tl.arange(0, BW)) < DW
    _mask_hw = _mask_h[:, None] & _mask_w[None, :]
    _for_C = min(DC - i_c * BC, BC)

    _tmp0 = i_c * BC * DH * DW
    _tmp1 = DC * DH * DW
    _tmp2 = (
        _tmp0
        + i_h * BH * DW
        + tl.arange(0, BH)[:, None] * DW
        + i_w * BW
        + tl.arange(0, BW)[None, :]
    )
    p_x = x + i_b * _tmp1 + _tmp2
    # p_y1 = y + i_b * 2 * _tmp1 + _tmp2  # same - 注意这里的索引从4变为2
    p_y2 = (
        y
        + i_b * _tmp1
        + _tmp0
        + i_w * BW * DH
        + tl.arange(0, BW)[None, :] * DH
        + i_h * BH
        + tl.arange(0, BH)[:, None]
    )  # trans

    for idxc in range(_for_C):
        _idx = idxc * DH * DW
        _x = tl.load(p_x + _idx, mask=_mask_hw)
        # tl.store(p_y1 + _idx, _x, mask=_mask_hw)
        tl.store(p_y2 + _idx, _x, mask=_mask_hw)
    tl.debug_barrier()


@triton.jit
def triton_cross_merge_trans(
    x,  # (B, C, H, W)
    y,  # (B, 2, C, H, W)
    BC: tl.constexpr,
    BH: tl.constexpr,
    BW: tl.constexpr,
    DC: tl.constexpr,
    DH: tl.constexpr,
    DW: tl.constexpr,
    NH: tl.constexpr,
    NW: tl.constexpr,
):
    i_hw, i_c, i_b = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_h, i_w = (i_hw // NW), (i_hw % NW)
    _mask_h = (i_h * BH + tl.arange(0, BH)) < DH
    _mask_w = (i_w * BW + tl.arange(0, BW)) < DW
    _mask_hw = _mask_h[:, None] & _mask_w[None, :]
    _for_C = min(DC - i_c * BC, BC)

    _tmp0 = i_c * BC * DH * DW
    _tmp1 = DC * DH * DW
    _tmp2 = (
        _tmp0
        + i_h * BH * DW
        + tl.arange(0, BH)[:, None] * DW
        + i_w * BW
        + tl.arange(0, BW)[None, :]
    )
    p_x = x + i_b * _tmp1 + _tmp2

    p_y2 = (
        y
        + i_b * _tmp1
        + _tmp0
        + i_w * BW * DH
        + tl.arange(0, BW)[None, :] * DH
        + i_h * BH
        + tl.arange(0, BH)[:, None]
    )

    for idxc in range(_for_C):
        _idx = idxc * DH * DW
        _y2 = tl.load(p_y2 + _idx, mask=_mask_hw)
        tl.store(p_x + _idx, _y2, mask=_mask_hw)
    tl.debug_barrier()


@triton.jit
def triton_cross_scan_trans_same(
    x,  # (B, C, H, W)
    y,  # (B, 1, C, H, W)
    BC: tl.constexpr,
    BH: tl.constexpr,
    BW: tl.constexpr,
    DC: tl.constexpr,
    DH: tl.constexpr,
    DW: tl.constexpr,
    NH: tl.constexpr,
    NW: tl.constexpr,
):
    i_hw, i_c, i_b = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_h, i_w = (i_hw // NW), (i_hw % NW)
    _mask_h = (i_h * BH + tl.arange(0, BH)) < DH
    _mask_w = (i_w * BW + tl.arange(0, BW)) < DW
    _mask_hw = _mask_h[:, None] & _mask_w[None, :]
    _for_C = min(DC - i_c * BC, BC)

    _tmp0 = i_c * BC * DH * DW
    _tmp1 = DC * DH * DW
    _tmp2 = (
        _tmp0
        + i_h * BH * DW
        + tl.arange(0, BH)[:, None] * DW
        + i_w * BW
        + tl.arange(0, BW)[None, :]
    )
    p_x = x + i_b * _tmp1 + _tmp2

    p_y3 = (
        y
        + i_b * _tmp1
        + _tmp0
        + (NH - i_h - 1) * BH * DW
        + (BH - 1 - tl.arange(0, BH)[:, None]) * DW
        + (NW - i_w - 1) * BW
        + (BW - 1 - tl.arange(0, BW)[None, :])
        + (DH - NH * BH) * DW
        + (DW - NW * BW)
    )  # flip
    # p_y4 = y + i_b * 2 * _tmp1 + _tmp1 + _tmp0 + (NW - i_w - 1) * BW * DH + (BW - 1 - tl.arange(0, BW)[None, :]) * DH + (NH - i_h - 1) * BH + (BH - 1 - tl.arange(0, BH)[:, None]) + (DH - NH * BH) + (DW - NW * BW) * DH  # trans + flip

    for idxc in range(_for_C):
        _idx = idxc * DH * DW
        _x = tl.load(p_x + _idx, mask=_mask_hw)
        # tl.store(p_y1 + _idx, _x, mask=_mask_hw)
        # tl.store(p_y2 + _idx, _x, mask=_mask_hw)
        tl.store(p_y3 + _idx, _x, mask=_mask_hw)
        # tl.store(p_y4 + _idx, _x, mask=_mask_hw)
    tl.debug_barrier()


@triton.jit
def triton_cross_merge_trans_same(
    x,  # (B, C, H, W)
    y,  # (B, 2, C, H, W)
    BC: tl.constexpr,
    BH: tl.constexpr,
    BW: tl.constexpr,
    DC: tl.constexpr,
    DH: tl.constexpr,
    DW: tl.constexpr,
    NH: tl.constexpr,
    NW: tl.constexpr,
):
    i_hw, i_c, i_b = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_h, i_w = (i_hw // NW), (i_hw % NW)
    _mask_h = (i_h * BH + tl.arange(0, BH)) < DH
    _mask_w = (i_w * BW + tl.arange(0, BW)) < DW
    _mask_hw = _mask_h[:, None] & _mask_w[None, :]
    _for_C = min(DC - i_c * BC, BC)

    _tmp0 = i_c * BC * DH * DW
    _tmp1 = DC * DH * DW
    _tmp2 = (
        _tmp0
        + i_h * BH * DW
        + tl.arange(0, BH)[:, None] * DW
        + i_w * BW
        + tl.arange(0, BW)[None, :]
    )
    p_x = x + i_b * _tmp1 + _tmp2

    p_y3 = (
        y
        + i_b * _tmp1
        + _tmp0
        + (NH - i_h - 1) * BH * DW
        + (BH - 1 - tl.arange(0, BH)[:, None]) * DW
        + (NW - i_w - 1) * BW
        + (BW - 1 - tl.arange(0, BW)[None, :])
        + (DH - NH * BH) * DW
        + (DW - NW * BW)
    )

    for idxc in range(_for_C):
        _idx = idxc * DH * DW
        _y3 = tl.load(p_y3 + _idx, mask=_mask_hw)
        tl.store(p_x + _idx, _y3, mask=_mask_hw)
    tl.debug_barrier()


@triton.jit
def triton_cross_scan_flips(
    x,  # (B, C, H, W)
    y,  # (B, 1, C, H, W)
    BC: tl.constexpr,
    BH: tl.constexpr,
    BW: tl.constexpr,
    DC: tl.constexpr,
    DH: tl.constexpr,
    DW: tl.constexpr,
    NH: tl.constexpr,
    NW: tl.constexpr,
):
    i_hw, i_c, i_b = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_h, i_w = (i_hw // NW), (i_hw % NW)
    _mask_h = (i_h * BH + tl.arange(0, BH)) < DH
    _mask_w = (i_w * BW + tl.arange(0, BW)) < DW
    _mask_hw = _mask_h[:, None] & _mask_w[None, :]
    _for_C = min(DC - i_c * BC, BC)

    _tmp0 = i_c * BC * DH * DW
    _tmp1 = DC * DH * DW
    _tmp2 = (
        _tmp0
        + i_h * BH * DW
        + tl.arange(0, BH)[:, None] * DW
        + i_w * BW
        + tl.arange(0, BW)[None, :]
    )
    p_x = x + i_b * _tmp1 + _tmp2

    # p_y3 = y + i_b * 2 * _tmp1 + _tmp0 + (NH - i_h - 1) * BH * DW + (BH - 1 - tl.arange(0, BH)[:, None]) * DW + (NW - i_w - 1) * BW + (BW - 1 - tl.arange(0, BW)[None, :]) + (DH - NH * BH) * DW + (DW - NW * BW)  # flip
    p_y4 = (
        y
        + i_b * _tmp1
        + _tmp0
        + (NW - i_w - 1) * BW * DH
        + (BW - 1 - tl.arange(0, BW)[None, :]) * DH
        + (NH - i_h - 1) * BH
        + (BH - 1 - tl.arange(0, BH)[:, None])
        + (DH - NH * BH)
        + (DW - NW * BW) * DH
    )  # trans + flip

    for idxc in range(_for_C):
        _idx = idxc * DH * DW
        _x = tl.load(p_x + _idx, mask=_mask_hw)
        # tl.store(p_y1 + _idx, _x, mask=_mask_hw)
        # tl.store(p_y2 + _idx, _x, mask=_mask_hw)
        # tl.store(p_y3 + _idx, _x, mask=_mask_hw)
        tl.store(p_y4 + _idx, _x, mask=_mask_hw)
    tl.debug_barrier()


@triton.jit
def triton_cross_merge_flips(
    x,  # (B, C, H, W)
    y,  # (B, 2, C, H, W)
    BC: tl.constexpr,
    BH: tl.constexpr,
    BW: tl.constexpr,
    DC: tl.constexpr,
    DH: tl.constexpr,
    DW: tl.constexpr,
    NH: tl.constexpr,
    NW: tl.constexpr,
):
    i_hw, i_c, i_b = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_h, i_w = (i_hw // NW), (i_hw % NW)
    _mask_h = (i_h * BH + tl.arange(0, BH)) < DH
    _mask_w = (i_w * BW + tl.arange(0, BW)) < DW
    _mask_hw = _mask_h[:, None] & _mask_w[None, :]
    _for_C = min(DC - i_c * BC, BC)

    _tmp0 = i_c * BC * DH * DW
    _tmp1 = DC * DH * DW
    _tmp2 = (
        _tmp0
        + i_h * BH * DW
        + tl.arange(0, BH)[:, None] * DW
        + i_w * BW
        + tl.arange(0, BW)[None, :]
    )
    p_x = x + i_b * _tmp1 + _tmp2

    p_y4 = (
        y
        + i_b * _tmp1
        + _tmp0
        + (NW - i_w - 1) * BW * DH
        + (BW - 1 - tl.arange(0, BW)[None, :]) * DH
        + (NH - i_h - 1) * BH
        + (BH - 1 - tl.arange(0, BH)[:, None])
        + (DH - NH * BH)
        + (DW - NW * BW) * DH
    )

    for idxc in range(_for_C):
        _idx = idxc * DH * DW
        _y4 = tl.load(p_y4 + _idx, mask=_mask_hw)
        tl.store(p_x + _idx, _y4, mask=_mask_hw)
    tl.debug_barrier()


############## class ###############
class CrossScanTriton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor):
        B, C, H, W = x.shape
        B, C, H, W = int(B), int(C), int(H), int(W)
        BC, BH, BW = (
            min(triton.next_power_of_2(C), 1),
            min(triton.next_power_of_2(H), 64),
            min(triton.next_power_of_2(W), 64),
        )
        NH, NW, NC = triton.cdiv(H, BH), triton.cdiv(W, BW), triton.cdiv(C, BC)
        ctx.shape = (B, C, H, W)
        ctx.triton_shape = (BC, BH, BW, NC, NH, NW)
        x = x.contiguous()
        y = x.new_empty((B, 4, C, H, W))
        triton_cross_scan[(NH * NW, NC, B)](x, y, BC, BH, BW, C, H, W, NH, NW)
        return y.view(B, 4, C, -1)

    @staticmethod
    def backward(ctx, y: torch.Tensor):
        # out: (b, k, d, l)
        B, C, H, W = ctx.shape
        BC, BH, BW, NC, NH, NW = ctx.triton_shape
        y = y.contiguous().view(B, 4, C, H, W)
        x = y.new_empty((B, C, H, W))
        triton_cross_merge[(NH * NW, NC, B)](x, y, BC, BH, BW, C, H, W, NH, NW)
        return x


class CrossMergeTriton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, y: torch.Tensor):
        B, K, C, H, W = y.shape
        B, C, H, W = int(B), int(C), int(H), int(W)
        BC, BH, BW = (
            min(triton.next_power_of_2(C), 1),
            min(triton.next_power_of_2(H), 64),
            min(triton.next_power_of_2(W), 64),
        )
        NH, NW, NC = triton.cdiv(H, BH), triton.cdiv(W, BW), triton.cdiv(C, BC)
        ctx.shape = (B, C, H, W)
        ctx.triton_shape = (BC, BH, BW, NC, NH, NW)
        y = y.contiguous().view(B, 4, C, H, W)
        x = y.new_empty((B, C, H, W))
        triton_cross_merge[(NH * NW, NC, B)](x, y, BC, BH, BW, C, H, W, NH, NW)
        return x.view(B, C, -1)

    @staticmethod
    def backward(ctx, x: torch.Tensor):
        # out: (b, d, l)
        B, C, H, W = ctx.shape
        BC, BH, BW, NC, NH, NW = ctx.triton_shape
        x = x.contiguous()
        y = x.new_empty((B, 4, C, H, W))
        triton_cross_scan[(NH * NW, NC, B)](x, y, BC, BH, BW, C, H, W, NH, NW)
        return y


class CrossScanTriton1b1(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor):
        B, K, C, H, W = x.shape
        B, C, H, W = int(B), int(C), int(H), int(W)
        BC, BH, BW = (
            min(triton.next_power_of_2(C), 1),
            min(triton.next_power_of_2(H), 64),
            min(triton.next_power_of_2(W), 64),
        )
        NH, NW, NC = triton.cdiv(H, BH), triton.cdiv(W, BW), triton.cdiv(C, BC)
        ctx.shape = (B, C, H, W)
        ctx.triton_shape = (BC, BH, BW, NC, NH, NW)
        x = x.contiguous()
        y = x.new_empty((B, 4, C, H, W))
        triton_cross_scan_1b1[(NH * NW, NC, B)](x, y, BC, BH, BW, C, H, W, NH, NW)
        return y.view(B, 4, C, -1)

    @staticmethod
    def backward(ctx, y: torch.Tensor):
        # out: (b, k, d, l)
        B, C, H, W = ctx.shape
        BC, BH, BW, NC, NH, NW = ctx.triton_shape
        y = y.contiguous().view(B, 4, C, H, W)
        x = y.new_empty((B, 4, C, H, W))
        triton_cross_merge_1b1[(NH * NW, NC, B)](x, y, BC, BH, BW, C, H, W, NH, NW)
        return x


class CrossScanTritonSelect(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, scan_mode: int = 0):
        ctx.scan_mode = scan_mode
        B, C, H, W = x.shape
        B, C, H, W = int(B), int(C), int(H), int(W)
        BC, BH, BW = (
            min(triton.next_power_of_2(C), 2),
            min(triton.next_power_of_2(H), 32),
            min(triton.next_power_of_2(W), 32),
        )
        NH, NW, NC = triton.cdiv(H, BH), triton.cdiv(W, BW), triton.cdiv(C, BC)
        ctx.shape = (B, C, H, W)
        ctx.triton_shape = (BC, BH, BW, NC, NH, NW)
        x = x.contiguous()
        y = x.new_empty((B, 2, C, H, W))
        if scan_mode == 0:
            triton_cross_scan_same_and_trans[(NH * NW, NC, B)](
                x, y, BC, BH, BW, C, H, W, NH, NW
            )
        elif scan_mode == 1:
            triton_cross_scan_trans_and_flips[(NH * NW, NC, B)](
                x, y, BC, BH, BW, C, H, W, NH, NW
            )
        else:
            raise RuntimeError("scan_mode should be 1 or 2")

        return y.view(B, 2, C, -1)

    @staticmethod
    def backward(ctx, y: torch.Tensor):
        # out: (b, k, d, l)
        B, C, H, W = ctx.shape
        BC, BH, BW, NC, NH, NW = ctx.triton_shape
        y = y.contiguous().view(B, 2, C, H, W)
        x = y.new_empty((B, C, H, W))
        if ctx.scan_mode == 0:
            triton_cross_merge_same_and_trans[(NH * NW, NC, B)](
                x, y, BC, BH, BW, C, H, W, NH, NW
            )
        elif ctx.scan_mode == 1:
            triton_cross_merge_trans_and_flips[(NH * NW, NC, B)](
                x, y, BC, BH, BW, C, H, W, NH, NW
            )
        else:
            raise RuntimeError("scan_mode should be 1 or 2")

        return x, None


class CrossMergeTritonSelect(torch.autograd.Function):
    @staticmethod
    def forward(ctx, y: torch.Tensor, scan_mode: int = 0):
        ctx.scan_mode = scan_mode
        B, K, C, H, W = y.shape
        B, C, H, W = int(B), int(C), int(H), int(W)
        BC, BH, BW = (
            min(triton.next_power_of_2(C), 2),
            min(triton.next_power_of_2(H), 32),
            min(triton.next_power_of_2(W), 32),
        )
        NH, NW, NC = triton.cdiv(H, BH), triton.cdiv(W, BW), triton.cdiv(C, BC)
        ctx.shape = (B, C, H, W)
        ctx.triton_shape = (BC, BH, BW, NC, NH, NW)
        y = y.contiguous().view(B, 2, C, H, W)
        x = y.new_empty((B, C, H, W))
        if scan_mode == 0:
            triton_cross_merge_same_and_trans[(NH * NW, NC, B)](
                x, y, BC, BH, BW, C, H, W, NH, NW
            )
        elif scan_mode == 1:
            triton_cross_merge_trans_and_flips[(NH * NW, NC, B)](
                x, y, BC, BH, BW, C, H, W, NH, NW
            )

        return x.view(B, C, -1)

    @staticmethod
    def backward(ctx, x: torch.Tensor):
        # out: (b, d, l)
        B, C, H, W = ctx.shape
        BC, BH, BW, NC, NH, NW = ctx.triton_shape
        x = x.contiguous()
        y = x.new_empty((B, 2, C, H, W))
        if ctx.scan_mode == 0:
            triton_cross_scan_same_and_trans[(NH * NW, NC, B)](
                x, y, BC, BH, BW, C, H, W, NH, NW
            )
        elif ctx.scan_mode == 1:
            triton_cross_scan_trans_and_flips[(NH * NW, NC, B)](
                x, y, BC, BH, BW, C, H, W, NH, NW
            )

        return y, None


class CrossScanTritonSelectK1(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, scan_mode: int = 0):
        ctx.scan_mode = scan_mode
        B, C, H, W = x.shape
        B, C, H, W = int(B), int(C), int(H), int(W)
        BC, BH, BW = (
            min(triton.next_power_of_2(C), 2),
            min(triton.next_power_of_2(H), 32),
            min(triton.next_power_of_2(W), 32),
        )
        NH, NW, NC = triton.cdiv(H, BH), triton.cdiv(W, BW), triton.cdiv(C, BC)
        ctx.shape = (B, C, H, W)
        ctx.triton_shape = (BC, BH, BW, NC, NH, NW)
        x = x.contiguous()
        y = x.new_empty((B, 1, C, H, W))
        if scan_mode == 0:
            triton_cross_scan_same[(NH * NW, NC, B)](x, y, BC, BH, BW, C, H, W, NH, NW)
        elif scan_mode == 1:
            triton_cross_scan_trans[(NH * NW, NC, B)](x, y, BC, BH, BW, C, H, W, NH, NW)
        elif scan_mode == 2:
            triton_cross_scan_trans_same[(NH * NW, NC, B)](
                x, y, BC, BH, BW, C, H, W, NH, NW
            )
        elif scan_mode == 3:
            triton_cross_scan_flips[(NH * NW, NC, B)](x, y, BC, BH, BW, C, H, W, NH, NW)
        else:
            raise RuntimeError("scan_mode should be 1 or 2")

        return y.view(B, 1, C, -1)

    @staticmethod
    def backward(ctx, y: torch.Tensor):
        # out: (b, k, d, l)
        B, C, H, W = ctx.shape
        BC, BH, BW, NC, NH, NW = ctx.triton_shape
        y = y.contiguous().view(B, 1, C, H, W)
        x = y.new_empty((B, C, H, W))
        if ctx.scan_mode == 0:
            triton_cross_merge_same[(NH * NW, NC, B)](x, y, BC, BH, BW, C, H, W, NH, NW)
        elif ctx.scan_mode == 1:
            triton_cross_merge_trans[(NH * NW, NC, B)](
                x, y, BC, BH, BW, C, H, W, NH, NW
            )
        elif ctx.scan_mode == 2:
            triton_cross_merge_trans_same[(NH * NW, NC, B)](
                x, y, BC, BH, BW, C, H, W, NH, NW
            )
        elif ctx.scan_mode == 3:
            triton_cross_merge_flips[(NH * NW, NC, B)](
                x, y, BC, BH, BW, C, H, W, NH, NW
            )
        else:
            raise RuntimeError("scan_mode should be 1 or 2")

        return x, None


class CrossMergeTritonSelectK1(torch.autograd.Function):
    @staticmethod
    def forward(ctx, y: torch.Tensor, scan_mode: int = 0):
        ctx.scan_mode = scan_mode
        B, K, C, H, W = y.shape
        B, C, H, W = int(B), int(C), int(H), int(W)
        BC, BH, BW = (
            min(triton.next_power_of_2(C), 2),
            min(triton.next_power_of_2(H), 32),
            min(triton.next_power_of_2(W), 32),
        )
        NH, NW, NC = triton.cdiv(H, BH), triton.cdiv(W, BW), triton.cdiv(C, BC)
        ctx.shape = (B, C, H, W)
        ctx.triton_shape = (BC, BH, BW, NC, NH, NW)
        y = y.contiguous().view(B, 1, C, H, W)
        x = y.new_empty((B, C, H, W))
        if scan_mode == 0:
            triton_cross_merge_same[(NH * NW, NC, B)](x, y, BC, BH, BW, C, H, W, NH, NW)
        elif scan_mode == 1:
            triton_cross_merge_trans[(NH * NW, NC, B)](
                x, y, BC, BH, BW, C, H, W, NH, NW
            )
        elif scan_mode == 2:
            triton_cross_merge_trans_same[(NH * NW, NC, B)](
                x, y, BC, BH, BW, C, H, W, NH, NW
            )
        elif scan_mode == 3:
            triton_cross_merge_flips[(NH * NW, NC, B)](
                x, y, BC, BH, BW, C, H, W, NH, NW
            )

        return x.view(B, C, -1)

    @staticmethod
    def backward(ctx, x: torch.Tensor):
        # out: (b, d, l)
        B, C, H, W = ctx.shape
        BC, BH, BW, NC, NH, NW = ctx.triton_shape
        x = x.contiguous()
        y = x.new_empty((B, 1, C, H, W))
        if ctx.scan_mode == 0:
            triton_cross_merge_same[(NH * NW, NC, B)](x, y, BC, BH, BW, C, H, W, NH, NW)
        elif ctx.scan_mode == 1:
            triton_cross_scan_trans[(NH * NW, NC, B)](x, y, BC, BH, BW, C, H, W, NH, NW)
        elif ctx.scan_mode == 2:
            triton_cross_scan_trans_same[(NH * NW, NC, B)](
                x, y, BC, BH, BW, C, H, W, NH, NW
            )
        elif ctx.scan_mode == 3:
            triton_cross_scan_flips[(NH * NW, NC, B)](x, y, BC, BH, BW, C, H, W, NH, NW)

        return y, None


if __name__ == "__main__":
    # torch.manual_seed(2024)
    # torch.cuda.manual_seed(2024)

    ## test Scanning operator

    # x = torch.randn(2, 3, 64, 64, dtype=torch.float16).cuda().requires_grad_() * 100

    y1 = CrossScanTritonSelect.apply(x, 0)
    grad = torch.autograd.grad(y1.sum(), x)[0]
    print(y1.shape)
    print(grad.shape)
    print("=========")
    y2 = CrossScanTritonSelect.apply(x, 1)
    grad = torch.autograd.grad(y2.sum(), x)[0]
    print(y2.shape)
    print(grad.shape)

    # y11 = CrossScanTritonSelectK1.apply(x, 0)
    # y22 = CrossScanTritonSelectK1.apply(x, 1)
    # y33 = CrossScanTritonSelectK1.apply(x, 2)
    # y44 = CrossScanTritonSelectK1.apply(x, 3)

    # yy = CrossScanTriton.apply(x)

    # print((y1 == yy[:, :2]).all())
    # print((y2 == yy[:, -2:]).all())
    # print((y11 == yy[:, :1]).all())
    # print((y22 == yy[:, 1:2]).all())
    # print((y33 == yy[:, 2:3]).all())
    # print((y44 == yy[:, 3:4]).all())

    # ## test Merge operator

    # y = torch.randn(2, 4, 3, 64, 64).cuda().requires_grad_() * 100
    # x = CrossMergeTriton.apply(y)

    # y = torch.randn(2, 2, 3, 64, 64, dtype=torch.float16).cuda().requires_grad_() * 100
    # x1 = CrossMergeTritonSelect.apply(y, 0)
    # grad = torch.autograd.grad(x1.sum(), y)[0]
    # print(x1.shape)
    # print(grad.shape)
    # print('=========')
    # x2 = CrossMergeTritonSelect.apply(y, 1)
    # grad = torch.autograd.grad(x2.sum(), y)[0]
    # print(x2.shape)
    # print(grad.shape)

    # x11 = CrossMergeTritonSelectK1.apply(y[:, 0:1], 0)
    # x22 = CrossMergeTritonSelectK1.apply(y[:, 1:2], 1)
    # x33 = CrossMergeTritonSelectK1.apply(y[:, 2:3], 2)
    # x44 = CrossMergeTritonSelectK1.apply(y[:, 3:4], 3)

    # print((x1 + x2 - x).max())
    # print((x11 + x22 + x33 + x44 - x).max())

    ## test K=1 torch

    # class CrossScanK1Torch(torch.nn.Module):
    #     def __init__(self,):
    #         super().__init__()

    #     def scan(self, x, scan_id):
    #         B, C, H, W = x.size()
    #         if scan_id == 0:
    #             return x.view(B, 1, C, -1)
    #         elif scan_id == 1:
    #             return x.transpose(dim0=2, dim1=3).flatten(2, 3)
    #         elif scan_id == 2:
    #             return torch.flip(x.view(B, 1, C, -1), dims=[-1])
    #         elif scan_id == 3:
    #             return torch.flip(x.transpose(dim0=2, dim1=3).flatten(2, 3), dims=[-1])

    #     def merge(self, x, scan_id):
    #         B, _, C, H, W = x.size()
    #         if scan_id == 0:
    #             return x.view(B, C, H, W)
    #         elif scan_id == 1:
    #             return x.view(B, C, W, H).transpose(dim0=2, dim1=3).contiguous().view(B, C, H, W)
    #         elif scan_id == 2:
    #             x = x.view(B, C, -1)
    #             return torch.flip(x, dims=[-1]).view(B, C, H, W)
    #         elif scan_id == 3:
    #             x = x.view(B, C, H, W)
    #             return torch.flip(x.view(B, C, W, H).transpose(dim0=2, dim1=3).flatten(2, 3), dims=[-1]).view(B, C, H, W)

    # scan_merge_fn = CrossScanK1Torch()

    # x = torch.randn(2, 3, 64, 64).cuda().requires_grad_() * 100

    # y1 = scan_merge_fn.scan(x, 0).view(2, 1, 3, 64, 64)
    # x1 = scan_merge_fn.merge(y1, 0).view(2, 3, 64, 64)

    # print((x1==x).all())

    # y2 = scan_merge_fn.scan(x, 1).view(2, 1, 3, 64, 64)
    # x2 = scan_merge_fn.merge(y2, 1).view(2, 3, 64, 64)

    # print((x2==x).all())

    # y3 = scan_merge_fn.scan(x, 2).view(2, 1, 3, 64, 64)
    # x3 = scan_merge_fn.merge(y3, 2).view(2, 3, 64, 64)

    # print((x3==x).all())

    # y4 = scan_merge_fn.scan(x, 0).view(2, 1, 3, 64, 64)
    # x4 = scan_merge_fn.merge(y4, 0).view(2, 3, 64, 64)

    # print((x4==x).all())

    ## test K1
    # x = torch.randn(2, 3, 64, 64).cuda().requires_grad_() * 100

    # x1 = scan_merge_fn.scan(x, 1)
    # x11 = CrossScanTritonSelectK1.apply(x, 1)

    # print((x1 - x11).abs().max())

    # for i in range(4):
    #     y = CrossScanTritonSelectK1.apply(x, i).view(2, 1, 3, 64, 64)
    #     x1 = CrossMergeTritonSelectK1.apply(y, i).view(2, 3, 64, 64)

    #     print((x==x1).all())
