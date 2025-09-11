# Copyright (c) Shanghai AI Lab. All rights reserved.
import math
import os
from functools import partial
from typing import Sequence

import torch
import torch.cuda.amp
import torch.nn as nn
import torch.utils.checkpoint as cp
import triton
from einops import rearrange
from einops.layers.torch import Rearrange
from timm.layers import DropPath
from torch.nn import functional as F
from torch.utils.cpp_extension import load

from model.module.csm_triton import (
    triton_cross_merge,
    triton_cross_merge_flips,
    triton_cross_merge_same,
    triton_cross_merge_same_and_trans,
    triton_cross_merge_trans,
    triton_cross_merge_trans_and_flips,
    triton_cross_merge_trans_same,
    triton_cross_scan,
    triton_cross_scan_flips,
    triton_cross_scan_same,
    triton_cross_scan_same_and_trans,
    triton_cross_scan_trans,
    triton_cross_scan_trans_and_flips,
    triton_cross_scan_trans_same,
)
from model.module.layer_norm import RMSNorm
from utils import easy_logger

logger = easy_logger(func_name="rwkv_v5_multi_modal")

############### K = 8 ################


def antidiagonal_gather(tensor):
    # 取出矩阵所有反斜向的元素并拼接
    B, C, H, W = tensor.size()
    shift = torch.arange(H, device=tensor.device).unsqueeze(1)  # 创建一个列向量[H, 1]
    index = (
        torch.arange(W, device=tensor.device) - shift
    ) % W  # 利用广播创建索引矩阵[H, W]
    # 扩展索引以适应B和C维度
    expanded_index = index.unsqueeze(0).unsqueeze(0).expand(B, C, -1, -1)
    # 使用gather进行索引选择
    return tensor.gather(3, expanded_index).transpose(-1, -2).reshape(B, C, H * W)


def diagonal_gather(tensor):
    # 取出矩阵所有反斜向的元素并拼接
    B, C, H, W = tensor.size()
    shift = torch.arange(H, device=tensor.device).unsqueeze(1)  # 创建一个列向量[H, 1]
    index = (
        shift + torch.arange(W, device=tensor.device)
    ) % W  # 利用广播创建索引矩阵[H, W]
    # 扩展索引以适应B和C维度
    expanded_index = index.unsqueeze(0).unsqueeze(0).expand(B, C, -1, -1)
    # 使用gather进行索引选择
    return tensor.gather(3, expanded_index).transpose(-1, -2).reshape(B, C, H * W)


def diagonal_scatter(tensor_flat, original_shape):
    # 把斜向元素拼接起来的一维向量还原为最初的矩阵形式
    B, C, H, W = original_shape
    shift = torch.arange(H, device=tensor_flat.device).unsqueeze(
        1
    )  # 创建一个列向量[H, 1]
    index = (
        shift + torch.arange(W, device=tensor_flat.device)
    ) % W  # 利用广播创建索引矩阵[H, W]
    # 扩展索引以适应B和C维度
    expanded_index = index.unsqueeze(0).unsqueeze(0).expand(B, C, -1, -1)
    # 创建一个空的张量来存储反向散布的结果
    result_tensor = torch.zeros(
        B, C, H, W, device=tensor_flat.device, dtype=tensor_flat.dtype
    )
    # 将平铺的张量重新变形为[B, C, H, W]，考虑到需要使用transpose将H和W调换
    tensor_reshaped = tensor_flat.reshape(B, C, W, H).transpose(-1, -2)
    # 使用scatter_根据expanded_index将元素放回原位
    result_tensor.scatter_(3, expanded_index, tensor_reshaped)
    return result_tensor


def antidiagonal_scatter(tensor_flat, original_shape):
    # 把反斜向元素拼接起来的一维向量还原为最初的矩阵形式
    B, C, H, W = original_shape
    shift = torch.arange(H, device=tensor_flat.device).unsqueeze(
        1
    )  # 创建一个列向量[H, 1]
    index = (
        torch.arange(W, device=tensor_flat.device) - shift
    ) % W  # 利用广播创建索引矩阵[H, W]
    expanded_index = index.unsqueeze(0).unsqueeze(0).expand(B, C, -1, -1)
    # 初始化一个与原始张量形状相同、元素全为0的张量
    result_tensor = torch.zeros(
        B, C, H, W, device=tensor_flat.device, dtype=tensor_flat.dtype
    )
    # 将平铺的张量重新变形为[B, C, W, H]，因为操作是沿最后一个维度收集的，需要调整形状并交换维度
    tensor_reshaped = tensor_flat.reshape(B, C, W, H).transpose(-1, -2)
    # 使用scatter_将元素根据索引放回原位
    result_tensor.scatter_(3, expanded_index, tensor_reshaped)
    return result_tensor


class CrossScan(torch.autograd.Function):
    # ZSJ 这里是把图像按照特定方向展平的地方，改变扫描方向可以在这里修改
    @staticmethod
    def forward(ctx, x: torch.Tensor):
        B, C, H, W = x.shape
        ctx.shape = (B, C, H, W)
        # xs = x.new_empty((B, 4, C, H * W))
        xs = x.new_empty((B, 8, C, H * W))
        # 添加横向和竖向的扫描
        xs[:, 0] = x.flatten(2, 3)
        xs[:, 1] = x.transpose(dim0=2, dim1=3).flatten(2, 3)
        xs[:, 2:4] = torch.flip(xs[:, 0:2], dims=[-1])

        # 提供斜向和反斜向的扫描
        xs[:, 4] = diagonal_gather(x)
        xs[:, 5] = antidiagonal_gather(x)
        xs[:, 6:8] = torch.flip(xs[:, 4:6], dims=[-1])

        return xs

    @staticmethod
    def backward(ctx, ys: torch.Tensor):
        # out: (b, k, d, l)
        B, C, H, W = ctx.shape
        L = H * W
        # 把横向和竖向的反向部分再反向回来，并和原来的横向和竖向相加
        # ys = ys[:, 0:2] + ys[:, 2:4].flip(dims=[-1]).view(B, 2, -1, L)
        y_rb = ys[:, 0:2] + ys[:, 2:4].flip(dims=[-1]).view(B, 2, -1, L)
        # 把竖向的部分转成横向，然后再相加,再转回最初是的矩阵形式
        # y = ys[:, 0] + ys[:, 1].view(B, -1, W, H).transpose(dim0=2, dim1=3).contiguous().view(B, -1, L)
        y_rb = y_rb[:, 0] + y_rb[:, 1].view(B, -1, W, H).transpose(
            dim0=2, dim1=3
        ).contiguous().view(B, -1, L)
        y_rb = y_rb.view(B, -1, H, W)

        # 把斜向和反斜向的反向部分再反向回来，并和原来的斜向和反斜向相加
        y_da = ys[:, 4:6] + ys[:, 6:8].flip(dims=[-1]).view(B, 2, -1, L)
        # 把斜向和反斜向的部分都转成原来的最初的矩阵形式，再相加
        y_da = diagonal_scatter(y_da[:, 0], (B, C, H, W)) + antidiagonal_scatter(
            y_da[:, 1], (B, C, H, W)
        )

        y_res = y_rb + y_da
        # return y.view(B, -1, H, W)
        return y_res


class CrossMerge(torch.autograd.Function):
    @staticmethod
    def forward(ctx, ys: torch.Tensor):
        B, K, D, H, W = ys.shape
        ctx.shape = (H, W)
        ys = ys.view(B, K, D, -1)
        # ys = ys[:, 0:2] + ys[:, 2:4].flip(dims=[-1]).view(B, 2, D, -1)
        # y = ys[:, 0] + ys[:, 1].view(B, -1, W, H).transpose(dim0=2, dim1=3).contiguous().view(B, D, -1)

        y_rb = ys[:, 0:2] + ys[:, 2:4].flip(dims=[-1]).view(B, 2, D, -1)
        # 把竖向的部分转成横向，然后再相加,再转回最初是的矩阵形式
        y_rb = y_rb[:, 0] + y_rb[:, 1].view(B, -1, W, H).transpose(
            dim0=2, dim1=3
        ).contiguous().view(B, D, -1)
        y_rb = y_rb.view(B, -1, H, W)

        # 把斜向和反斜向的反向部分再反向回来，并和原来的斜向和反斜向相加
        y_da = ys[:, 4:6] + ys[:, 6:8].flip(dims=[-1]).view(B, 2, D, -1)
        # 把斜向和反斜向的部分都转成原来的最初的矩阵形式，再相加
        y_da = diagonal_scatter(y_da[:, 0], (B, D, H, W)) + antidiagonal_scatter(
            y_da[:, 1], (B, D, H, W)
        )

        y_res = y_rb + y_da
        return y_res.view(B, D, -1)
        # return y

    @staticmethod
    def backward(ctx, x: torch.Tensor):
        # B, D, L = x.shape
        # out: (b, k, d, l)
        H, W = ctx.shape
        B, C, L = x.shape
        # xs = x.new_empty((B, 4, C, L))
        xs = x.new_empty((B, 8, C, L))

        # 横向和竖向扫描
        xs[:, 0] = x
        xs[:, 1] = x.view(B, C, H, W).transpose(dim0=2, dim1=3).flatten(2, 3)
        xs[:, 2:4] = torch.flip(xs[:, 0:2], dims=[-1])
        # xs = xs.view(B, 4, C, H, W)

        # 提供斜向和反斜向的扫描
        xs[:, 4] = diagonal_gather(x.view(B, C, H, W))
        xs[:, 5] = antidiagonal_gather(x.view(B, C, H, W))
        xs[:, 6:8] = torch.flip(xs[:, 4:6], dims=[-1])

        # return xs
        return xs.view(B, 8, C, H, W)


############### K = 4 ################


class CrossScanTriton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor):
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
        y = x.new_empty((B, 4, C, H, W))
        triton_cross_scan[(NH * NW, NC, B)](x, y, BC, BH, BW, C, H, W, NH, NW)
        return y  # .view(B, 4, C, -1)

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
            min(triton.next_power_of_2(C), 2),
            min(triton.next_power_of_2(H), 32),
            min(triton.next_power_of_2(W), 32),
        )
        NH, NW, NC = triton.cdiv(H, BH), triton.cdiv(W, BW), triton.cdiv(C, BC)
        ctx.shape = (B, C, H, W)
        ctx.triton_shape = (BC, BH, BW, NC, NH, NW)
        y = y.contiguous().view(B, 4, C, H, W)
        x = y.new_empty((B, C, H, W))
        triton_cross_merge[(NH * NW, NC, B)](x, y, BC, BH, BW, C, H, W, NH, NW)
        return x  # .view(B, C, -1)

    @staticmethod
    def backward(ctx, x: torch.Tensor):
        # out: (b, d, l)
        B, C, H, W = ctx.shape
        BC, BH, BW, NC, NH, NW = ctx.triton_shape
        x = x.contiguous()
        y = x.new_empty((B, 4, C, H, W))
        triton_cross_scan[(NH * NW, NC, B)](x, y, BC, BH, BW, C, H, W, NH, NW)
        return y


############## K = 2 ##################


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

        return y  # .view(B, 2, C, -1)

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

        return x  # .view(B, C, -1)

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


############ K = 1 ################


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
            raise RuntimeError("scan_mode should be 0, 1, 2 or 3")

        return y  # .view(B, 1, C, -1)

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
            raise RuntimeError("scan_mode should be 0, 1, 2 or 3")

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

        return x  # .view(B, C, -1)

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


class CrossScanK1Torch:
    @staticmethod
    def scan(x, scan_id):
        B, C, H, W = x.size()
        if scan_id == 0:
            return x.view(B, 1, C, -1)
        elif scan_id == 1:
            return x.transpose(dim0=2, dim1=3).flatten(2, 3)
        elif scan_id == 2:
            return torch.flip(x.view(B, 1, C, -1), dims=[-1])
        elif scan_id == 3:
            return torch.flip(x.transpose(dim0=2, dim1=3).flatten(2, 3), dims=[-1])

    @staticmethod
    def merge(x, scan_id):
        B, _, C, H, W = x.size()
        if scan_id == 0:
            return x.view(B, C, H, W)
        elif scan_id == 1:
            return (
                x.view(B, C, W, H)
                .transpose(dim0=2, dim1=3)
                .contiguous()
                .view(B, C, H, W)
            )
        elif scan_id == 2:
            x = x.view(B, C, -1)
            return torch.flip(x, dims=[-1]).view(B, C, H, W)
        elif scan_id == 3:
            x = x.view(B, C, H, W)
            return torch.flip(
                x.view(B, C, W, H).transpose(dim0=2, dim1=3).flatten(2, 3), dims=[-1]
            ).view(B, C, H, W)


################# K=0 ################

# do not change x
# only flatten

###! just to be aligned with K=1, 2, 4, 8
#! the rearrange is not necessary, just for alignment


def cross_scan_k0(x):
    B, C, H, W = x.size()

    return x.view(B, 1, C, H, W)


def cross_merge_k0(x):
    B, _, C, H, W = x.size()
    x = x.squeeze(1)  # k to be 1
    return x  # .view(B, C, -1)


###################################### RWKV ############################################

wkv_version = os.getenv("WKV_VERSION", "5_x052")
logger.info(f"WKV_VERSION is set to {wkv_version}")

logger.info("-" * 80)
if wkv_version == "5":
    _TMAX_DEFAULT = 256 * 256
    T_MAX = int(os.getenv("T_MAX", _TMAX_DEFAULT))
    if T_MAX > _TMAX_DEFAULT:
        logger.warning(
            f"T_MAX is set to {T_MAX}, which is greater than {_TMAX_DEFAULT} by default. This may consume a lot of memory"
        )
    else:
        logger.info(f"T_MAX is set to {T_MAX}")
    HEAD_SIZE = 32  # not used

    CUDA_PATH = f"{os.path.dirname(__file__)}/rwkv_cuda/wkv5_cuda_re.cu"
    CPP_PATH = f"{os.path.dirname(__file__)}/rwkv_cuda/wkv5_op_re.cpp"
    BUILD_PATH = f"{os.path.dirname(__file__)}/rwkv_cuda/build_wkv5_{T_MAX}/"
    os.makedirs(BUILD_PATH, exist_ok=True)

    logger.info(
        "loading CUDA extension for [green]Vision RWKV5 Multi-Modal Operator[/green]🎉"
    )
    logger.info(f"with configure: [g] T_MAX={T_MAX}[/g]")
    wkv5_cuda = load(
        name="wkv",
        sources=[CPP_PATH, CUDA_PATH],
        verbose=True,
        extra_cuda_cflags=[
            "-res-usage",
            "--maxrregcount 60",
            "--use_fast_math",
            "-O3",
            "-Xptxas -O3",
            f"-DTmax={T_MAX}",
        ],
        build_directory=BUILD_PATH,
    )
elif wkv_version == "5_x052":
    HEAD_SIZE = 32
    CUDA_PATH = f"{os.path.dirname(__file__)}/rwkv_cuda/wkv5_cuda.cu"
    CPP_PATH = f"{os.path.dirname(__file__)}/rwkv_cuda/wkv5_op.cpp"
    BUILD_PATH = (
        f"{os.path.dirname(__file__)}/rwkv_cuda/build_wkv5_x052_head_{HEAD_SIZE}/"
    )
    os.makedirs(BUILD_PATH, exist_ok=True)
    wkv5_x052_cuda = load(
        name="wkv5",
        sources=[CPP_PATH, CUDA_PATH],
        verbose=True,
        extra_cuda_cflags=[
            "-res-usage",
            "--use_fast_math",
            "-O3",
            "-Xptxas -O3",
            "--extra-device-vectorization",
            f"-D_N_={HEAD_SIZE}",
        ],
        build_directory=BUILD_PATH,
    )
elif wkv_version == "6":
    _TMAX_DEFAULT = 256 * 256
    T_MAX = int(os.getenv("T_MAX", _TMAX_DEFAULT))
    if T_MAX > _TMAX_DEFAULT:
        logger.warning(
            f"T_MAX is set to {T_MAX}, which is greater than {_TMAX_DEFAULT} by default. This may consume a lot of memory"
        )
    else:
        logger.info(f"T_MAX is set to {T_MAX}")
    HEAD_SIZE = int(os.getenv("HEAD_SIZE", 32))
    TIME_DECAY_DIM = int(os.getenv("TIME_DECAY_DIM", 64))  # previous value: 64
    # ver1: vision RWKV
    # CUDA_PATH = f"{os.path.dirname(__file__)}/rwkv_cuda/wkv6_cuda_vrwkv.cu"
    # CPP_PATH = f"{os.path.dirname(__file__)}/rwkv_cuda/wkv6_op_vrwkv.cpp"

    # ver2: original RWKV
    CUDA_PATH = f"{os.path.dirname(__file__)}/rwkv_cuda/wkv6_cuda.cu"
    CPP_PATH = f"{os.path.dirname(__file__)}/rwkv_cuda/wkv6_op.cpp"
    BUILD_PATH = f"{os.path.dirname(__file__)}/rwkv_cuda/build_wkv6_{T_MAX}/"
    os.makedirs(BUILD_PATH, exist_ok=True)

    logger.info(
        "loading CUDA extension for [green]Vision RWKV6 Multi-Modal Operator[/green]🎉"
    )
    logger.info(
        f"with configure: [g] HEAD_SIZE={HEAD_SIZE}, TIME_DECAY_DIM={TIME_DECAY_DIM}, T_MAX={T_MAX}[/g]"
    )
    wkv6_cuda = load(
        name="wkv6",
        sources=[CPP_PATH, CUDA_PATH],
        verbose=True,
        extra_cuda_cflags=[
            "-res-usage",
            "--use_fast_math",
            "-O3",
            "-Xptxas -O3",
            "--extra-device-vectorization",
            f"-D_N_={HEAD_SIZE}",
            f"-D_T_={T_MAX}",
        ],
        build_directory=BUILD_PATH,
    )
elif wkv_version == "7":
    HEAD_SIZE = 32
    CHUNK_LEN = 16
    T_MAX = 8192
    HEAD_SIZE = int(os.getenv("HEAD_SIZE", 32))
    TIME_DECAY_DIM = int(os.getenv("TIME_DECAY_DIM", 64))  # previous value: 64

    CUDA_PATH = f"{os.path.dirname(__file__)}/rwkv_cuda/wkv7.cu"
    CPP_PATH = f"{os.path.dirname(__file__)}/rwkv_cuda/wkv7_op.cpp"
    BUILD_PATH = f"{os.path.dirname(__file__)}/rwkv_cuda/build_wkv7_{T_MAX}/"
    os.makedirs(BUILD_PATH, exist_ok=True)

    logger.info(
        "loading CUDA extension for [green]Vision RWKV7 Multi-Modal Operator[/green]🎉"
    )
    logger.info(
        f"with configure: [g]HEAD_SIZE={HEAD_SIZE}, TIME_DECAY_DIM={TIME_DECAY_DIM}, T_MAX={T_MAX}[/g]"
    )
    wkv7_cuda = load(
        name="wkv7",
        sources=[CPP_PATH, CUDA_PATH],
        # is_python_module=False,
        verbose=True,
        extra_cuda_cflags=[
            "-res-usage",
            "--use_fast_math",
            "-O3",
            "-Xptxas -O3",
            "--extra-device-vectorization",
            f"-D_N_={HEAD_SIZE}",
            f"-D_T_={T_MAX}",
            f"-D_CHUNK_LEN_={CHUNK_LEN}",
        ],
        build_directory=BUILD_PATH,
    )

logger.info(f"-" * 80)

################################################## RWKV  ###################################################

############################################### RWKV 7 ###############################################
DTYPE = torch.float32
XTYPE = torch.float32


class WKV_7g(torch.autograd.Function):
    @staticmethod
    def forward(ctx, r, w, k, v, a, b):
        with torch.no_grad():
            B, T, C = r.size()
            H = C // HEAD_SIZE
            N = HEAD_SIZE
            A = T // CHUNK_LEN
            assert HEAD_SIZE == C // H
            assert T % CHUNK_LEN == 0
            assert r.dtype == DTYPE
            assert w.dtype == DTYPE
            assert k.dtype == DTYPE
            assert v.dtype == DTYPE
            assert a.dtype == DTYPE
            assert b.dtype == DTYPE
            assert r.is_contiguous()
            assert w.is_contiguous()
            assert k.is_contiguous()
            assert v.is_contiguous()
            assert a.is_contiguous()
            assert b.is_contiguous()
            ctx.B = B
            ctx.T = T
            ctx.C = C
            ctx.H = H
            y = torch.empty(
                (B, T, C),
                device=k.device,
                dtype=DTYPE,
                memory_format=torch.contiguous_format,
            )  # .uniform_(-100, 100)
            saa = torch.empty(
                (B, T, H, N),
                device=k.device,
                dtype=torch.float,
                memory_format=torch.contiguous_format,
            )  # .uniform_(-100, 100)
            sss = torch.empty(
                (B, H, A, N, N),
                device=k.device,
                dtype=torch.float,
                memory_format=torch.contiguous_format,
            )  # .uniform_(-100, 100)
            # torch.ops.wkv7.forward(B, T, C, H, r, w, k, v, a, b, y, saa, sss)
            wkv7_cuda.forward(B, T, C, H, r, w, k, v, a, b, y, saa, sss)
            ctx.save_for_backward(r, w, k, v, a, b, saa, sss)
            return y

    @staticmethod
    def backward(ctx, gy):
        with torch.no_grad():
            N = HEAD_SIZE
            B = ctx.B
            T = ctx.T
            C = ctx.C
            H = ctx.H
            A = T // CHUNK_LEN
            assert gy.dtype == DTYPE
            assert gy.is_contiguous()
            r, w, k, v, a, b, saa, sss = ctx.saved_tensors
            gr = torch.empty(
                (B, T, C),
                device=gy.device,
                requires_grad=False,
                dtype=DTYPE,
                memory_format=torch.contiguous_format,
            )  # .zero_()#.uniform_(-100, 100)
            gw = torch.empty(
                (B, T, C),
                device=gy.device,
                requires_grad=False,
                dtype=DTYPE,
                memory_format=torch.contiguous_format,
            )  # .zero_()#.uniform_(-100, 100)
            gk = torch.empty(
                (B, T, C),
                device=gy.device,
                requires_grad=False,
                dtype=DTYPE,
                memory_format=torch.contiguous_format,
            )  # .zero_()#.uniform_(-100, 100)
            gv = torch.empty(
                (B, T, C),
                device=gy.device,
                requires_grad=False,
                dtype=DTYPE,
                memory_format=torch.contiguous_format,
            )  # .zero_()#.uniform_(-100, 100)
            ga = torch.empty(
                (B, T, C),
                device=gy.device,
                requires_grad=False,
                dtype=DTYPE,
                memory_format=torch.contiguous_format,
            )  # .uniform_(-100, 100)
            gb = torch.empty(
                (B, T, C),
                device=gy.device,
                requires_grad=False,
                dtype=DTYPE,
                memory_format=torch.contiguous_format,
            )  # .uniform_(-100, 100)
            zzz = torch.empty(
                (B, H, A - 1, N, N),
                device=gy.device,
                dtype=XTYPE,
                memory_format=torch.contiguous_format,
            )  # .uniform_(-100, 100)
            # torch.ops.wkv7.backward(B, T, C, H, r, w, k, v, a, b, saa, sss, zzz, gy, gr, gw, gk, gv, ga, gb)
            wkv7_cuda.backward(
                B, T, C, H, r, w, k, v, a, b, saa, sss, zzz, gy, gr, gw, gk, gv, ga, gb
            )
            del saa
            del sss
            del zzz
            return (gr, gw, gk, gv, ga, gb)


@torch.compiler.disable
def RUN_CUDA_RWKV7g(r, w, k, v, a, b):
    return WKV_7g.apply(r, w, k, v, a, b)


################################################## RWKV 6 ###################################################


class WKV_6(torch.autograd.Function):
    @staticmethod
    # @torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, B, T, C, H, r, k, v, w, u):
        with torch.no_grad():
            assert HEAD_SIZE == C // H
            ctx.B = B
            ctx.T = T
            ctx.C = C
            ctx.H = H
            assert r.is_contiguous()
            assert k.is_contiguous()
            assert v.is_contiguous()
            assert w.is_contiguous()
            assert u.is_contiguous()
            ew = (-torch.exp(w.float())).contiguous()
            ctx.save_for_backward(r, k, v, ew, u)
            y = torch.empty(
                (B, T, C),
                device=r.device,
                dtype=torch.float32,
                memory_format=torch.contiguous_format,
            )  # .uniform_(-100, 100)
            wkv6_cuda.forward(B, T, C, H, r, k, v, ew, u, y)
            return y

    @staticmethod
    # @torch.cuda.amp.custom_bwd
    def backward(ctx, gy):
        with torch.no_grad():
            B = ctx.B
            T = ctx.T
            C = ctx.C
            H = ctx.H
            if not gy.is_contiguous():
                gy = gy.contiguous()
            # assert gy.is_contiguous()
            r, k, v, ew, u = ctx.saved_tensors
            gr = torch.empty(
                (B, T, C),
                device=gy.device,
                requires_grad=False,
                dtype=torch.float32,
                memory_format=torch.contiguous_format,
            )  # .uniform_(-100, 100)
            gk = torch.empty(
                (B, T, C),
                device=gy.device,
                requires_grad=False,
                dtype=torch.float32,
                memory_format=torch.contiguous_format,
            )  # .uniform_(-100, 100)
            gv = torch.empty(
                (B, T, C),
                device=gy.device,
                requires_grad=False,
                dtype=torch.float32,
                memory_format=torch.contiguous_format,
            )  # .uniform_(-100, 100)
            gw = torch.empty(
                (B, T, C),
                device=gy.device,
                requires_grad=False,
                dtype=torch.float32,
                memory_format=torch.contiguous_format,
            )  # .uniform_(-100, 100)
            gu = torch.empty(
                (B, C),
                device=gy.device,
                requires_grad=False,
                dtype=torch.float32,
                memory_format=torch.contiguous_format,
            )  # .uniform_(-100, 100)
            wkv6_cuda.backward(B, T, C, H, r, k, v, ew, u, gy, gr, gk, gv, gw, gu)
            gu = torch.sum(gu, 0).view(H, C // H)
            return (None, None, None, None, gr, gk, gv, gw, gu)


def RUN_CUDA_RWKV6(B, T, C, H, r, k, v, w, u):
    # B = B.item()
    # T = T.item()
    # C = C.item()
    return WKV_6.apply(B, T, C, H, r, k, v, w, u)


################################################## RWKV 5 ###################################################


class WKV_5(torch.autograd.Function):
    @staticmethod
    def forward(ctx, B, T, C, w, u, k, v):
        ctx.B = B
        ctx.T = T
        ctx.C = C
        # assert T <= T_MAX, f'T={T}, T_MAX={T_MAX}, T must be less than T_MAX'
        assert (
            B * C % min(C, 1024) == 0
        ), f"B={B}, C={C}, B*C must be divisible by min(C, 1024)"

        half_mode = w.dtype == torch.half
        bf_mode = w.dtype == torch.bfloat16
        ctx.save_for_backward(w, u, k, v)
        w = w.float().contiguous()
        u = u.float().contiguous()
        k = k.float().contiguous()
        v = v.float().contiguous()
        y = torch.empty((B, T, C), device="cuda", memory_format=torch.contiguous_format)
        wkv5_cuda.forward(B, T, C, w, u, k, v, y)
        if half_mode:
            y = y.half()
        elif bf_mode:
            y = y.bfloat16()
        return y

    @staticmethod
    def backward(ctx, gy):
        B = ctx.B
        T = ctx.T
        C = ctx.C
        assert (
            T <= T_MAX
        ), f"for backward, T={T}, T_MAX={T_MAX}, T must be less than T_MAX"
        assert (
            B * C % min(C, 1024) == 0
        ), f"for backward, B={B}, C={C}, B*C must be divisible by min(C, 1024)"
        w, u, k, v = ctx.saved_tensors
        gw = torch.zeros((B, C), device="cuda").contiguous()
        gu = torch.zeros((B, C), device="cuda").contiguous()
        gk = torch.zeros((B, T, C), device="cuda").contiguous()
        gv = torch.zeros((B, T, C), device="cuda").contiguous()
        half_mode = w.dtype == torch.half
        bf_mode = w.dtype == torch.bfloat16
        wkv5_cuda.backward(
            B,
            T,
            C,
            w.float().contiguous(),
            u.float().contiguous(),
            k.float().contiguous(),
            v.float().contiguous(),
            gy.float().contiguous(),
            gw,
            gu,
            gk,
            gv,
        )
        if half_mode:
            gw = torch.sum(gw.half(), dim=0)
            gu = torch.sum(gu.half(), dim=0)
            return (None, None, None, gw.half(), gu.half(), gk.half(), gv.half())
        elif bf_mode:
            gw = torch.sum(gw.bfloat16(), dim=0)
            gu = torch.sum(gu.bfloat16(), dim=0)
            return (
                None,
                None,
                None,
                gw.bfloat16(),
                gu.bfloat16(),
                gk.bfloat16(),
                gv.bfloat16(),
            )
        else:
            gw = torch.sum(gw, dim=0)
            gu = torch.sum(gu, dim=0)
            return (None, None, None, gw, gu, gk, gv)


def RUN_CUDA_RWKV5(B, T, C, w, u, k, v):
    return WKV_5.apply(B, T, C, w.cuda(), u.cuda(), k.cuda(), v.cuda())


################################################## RWKV 5.2 ###################################################


class WKV_5_2(torch.autograd.Function):
    @staticmethod
    def forward(ctx, B, T, C, H, r, k, v, w, u):
        with torch.no_grad():
            assert (
                HEAD_SIZE == C // H
            ), f"HEAD_SIZE={HEAD_SIZE}, C={C}, H={H}, C//H = {C//H} != HEAD_SIZE"
            ctx.B = B
            ctx.T = T
            ctx.C = C
            ctx.H = H
            assert r.is_contiguous()
            assert k.is_contiguous()
            assert v.is_contiguous()
            assert w.is_contiguous()
            assert u.is_contiguous()
            ew = (-torch.exp(w.float())).contiguous()
            eew = (torch.exp(ew)).contiguous()
            ctx.save_for_backward(r, k, v, eew, ew, u)
            y = torch.empty(
                (B, T, C),
                device=r.device,
                dtype=torch.float32,
                memory_format=torch.contiguous_format,
            )  # .uniform_(-1, 1)
            wkv5_x052_cuda.forward(B, T, C, H, r, k, v, eew, u, y)
            return y

    @staticmethod
    def backward(ctx, gy):
        with torch.no_grad():
            assert gy.dtype == torch.float32
            B = ctx.B
            T = ctx.T
            C = ctx.C
            H = ctx.H
            assert gy.is_contiguous()
            r, k, v, eew, ew, u = ctx.saved_tensors
            gr = torch.empty(
                (B, T, C),
                device=gy.device,
                requires_grad=False,
                dtype=torch.float32,
                memory_format=torch.contiguous_format,
            )  # .uniform_(-1, 1)
            gk = torch.empty(
                (B, T, C),
                device=gy.device,
                requires_grad=False,
                dtype=torch.float32,
                memory_format=torch.contiguous_format,
            )  # .uniform_(-1, 1)
            gv = torch.empty(
                (B, T, C),
                device=gy.device,
                requires_grad=False,
                dtype=torch.float32,
                memory_format=torch.contiguous_format,
            )  # .uniform_(-1, 1)
            gw = torch.empty(
                (B, C),
                device=gy.device,
                requires_grad=False,
                dtype=torch.float32,
                memory_format=torch.contiguous_format,
            )  # .uniform_(-1, 1)
            gu = torch.empty(
                (B, C),
                device=gy.device,
                requires_grad=False,
                dtype=torch.float32,
                memory_format=torch.contiguous_format,
            )  # .uniform_(-1, 1)
            wkv5_x052_cuda.backward(
                B, T, C, H, r, k, v, eew, ew, u, gy, gr, gk, gv, gw, gu
            )
            gw = torch.sum(gw, 0).view(H, C // H)
            gu = torch.sum(gu, 0).view(H, C // H)
            return (None, None, None, None, gr, gk, gv, gw, gu)


def RUN_CUDA_RWKV5_2(B, T, C, H, r, k, v, w, u):
    # B = B.item()
    # T = T.item()
    # C = C.item()
    return WKV_5_2.apply(B, T, C, H, r, k, v, w, u)


################################################## Shift ###################################################


def q_shift_multihead(
    input,
    shift_pixel=1,
    head_dim=32,  # HEAD_SIZE,
    patch_resolution=None,
    with_cls_token=False,
):
    B, C, N = input.shape
    assert C % head_dim == 0
    assert head_dim % 4 == 0
    if with_cls_token:
        cls_tokens = input[:, [-1], :]
        input = input[:, :-1, :]
    input = input.reshape(
        B, -1, head_dim, patch_resolution[0], patch_resolution[1]
    )  # [B, n_head, head_dim H, W]
    B, _, _, H, W = input.shape
    output = torch.zeros_like(input)
    output[:, :, 0 : int(head_dim * 1 / 4), :, shift_pixel:W] = input[
        :, :, 0 : int(head_dim * 1 / 4), :, 0 : W - shift_pixel
    ]
    output[:, :, int(head_dim / 4) : int(head_dim / 2), :, 0 : W - shift_pixel] = input[
        :, :, int(head_dim / 4) : int(head_dim / 2), :, shift_pixel:W
    ]
    output[:, :, int(head_dim / 2) : int(head_dim / 4 * 3), shift_pixel:H, :] = input[
        :, :, int(head_dim / 2) : int(head_dim / 4 * 3), 0 : H - shift_pixel, :
    ]
    output[:, :, int(head_dim * 3 / 4) : int(head_dim), 0 : H - shift_pixel, :] = input[
        :, :, int(head_dim * 3 / 4) : int(head_dim), shift_pixel:H, :
    ]
    if with_cls_token:
        output = output.reshape(B, C, N - 1)  # .transpose(1, 2)
        output = torch.cat((output, cls_tokens), dim=1)
    else:
        output = output.reshape(B, C, N)  # .transpose(1, 2)
    return output


def groups_q_shift(
    input: torch.Tensor,
    shift_pixel: int = 1,
    gamma: float = 1 / 4,
    H: int = 64,
    W: int = 64,
    K: int = 1,
):
    assert gamma <= 1 / 4
    B, KC, N = input.size()
    C = KC // K
    # input = input.reshape(B, K, C, H, W)
    input = rearrange(input, "b (h w) c -> b c h w")
    output = torch.zeros_like(input)
    output[..., 0 : int(C * gamma), :, shift_pixel:W] = input[
        ..., 0 : int(C * gamma), :, 0 : W - shift_pixel
    ]
    output[..., int(C * gamma) : int(C * gamma * 2), :, 0 : W - shift_pixel] = input[
        ..., int(C * gamma) : int(C * gamma * 2), :, shift_pixel:W
    ]
    output[..., int(C * gamma * 2) : int(C * gamma * 3), shift_pixel:H, :] = input[
        ..., int(C * gamma * 2) : int(C * gamma * 3), 0 : H - shift_pixel, :
    ]
    output[..., int(C * gamma * 3) : int(C * gamma * 4), 0 : H - shift_pixel, :] = (
        input[..., int(C * gamma * 3) : int(C * gamma * 4), shift_pixel:H, :]
    )
    output[..., int(C * gamma * 4) :, :, :] = input[..., int(C * gamma * 4) :, :, :]
    # output = output.flatten(3)
    # output = output.view(B, KC, H*W)
    output = rearrange(output, "b c h w -> b (h w) c")
    return output


class ShiftByConv(nn.Module):
    def __init__(
        self,
        chan,
    ):
        super().__init__()
        self.dwconv = nn.Conv2d(
            chan,
            chan,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=chan,
            padding_mode="reflect",
        )

    def forward(self, input, H, W):
        input = input.permute(0, -1, 1, 2)
        bs, c = input.shape[:2]
        input = input.view(bs, c, H, W)
        input = self.dwconv(input)
        input = input.view(bs, c, -1)

        return input


################################################# Vision RWKV v6 and v7 Module ###################################################


class VRWKV_SpatialMix_V7(nn.Module):
    def __init__(
        self,
        n_embd,
        n_head,
        n_layer,
        layer_id,
        shift_mode="none",
        shift_pixel=1,
        attn_bias=False,
        init_mode="fancy",
        key_norm=False,
        with_cls_token=False,
        with_cp=False,
        img_txt_cat_order=0,
        scan_K=1,
    ):
        super().__init__()
        self.layer_id = layer_id
        self.n_embd = n_embd
        self.dim_att = dim_att = n_embd
        self.shift_mode = shift_mode
        self.shift_pixel = shift_pixel
        self.with_cls_token = with_cls_token
        self.img_txt_cat_order = img_txt_cat_order
        self.scan_K = scan_K
        self.with_cp = with_cp

        self.head_size = head_size = HEAD_SIZE
        self.n_head = n_head = self.dim_att // head_size
        assert self.dim_att % n_head == 0

        with torch.no_grad():
            ratio_0_to_1 = layer_id / (n_layer - 1)  # 0 to 1
            ratio_1_to_almost0 = 1.0 - (layer_id / n_layer)  # 1 to ~0
            ddd = torch.ones(1, 1, n_embd)
            for i in range(n_embd):
                ddd[0, 0, i] = i / n_embd

            # initialization comes from fitting my RWKV-6 7B runs
            # merging r&g w&a to save params
            self.time_maa_x = nn.Parameter(
                1.0 - torch.pow(ddd, 0.6 * ratio_1_to_almost0**0.9)
            )
            self.time_maa_rg = nn.Parameter(
                1.0 - torch.pow(ddd, 0.2 * ratio_1_to_almost0)
            )
            self.time_maa_wa = nn.Parameter(
                1.0 - torch.pow(ddd, 0.9 * ratio_1_to_almost0)
            )
            self.time_maa_k = nn.Parameter(
                1.0 - (torch.pow(ddd, 0.9 * ratio_1_to_almost0) + 0.4 * ratio_0_to_1)
            )
            self.time_maa_v = nn.Parameter(
                1.0 - (torch.pow(ddd, 0.4 * ratio_1_to_almost0) + 0.6 * ratio_0_to_1)
            )

            decay_speed = torch.ones(dim_att)
            for n in range(dim_att):
                decay_speed[n] = -7 + 5 * (n / (dim_att - 1)) ** (
                    0.85 + 1.0 * ratio_0_to_1**0.5
                )
            self.time_decay = nn.Parameter(
                decay_speed.reshape(1, 1, dim_att) + 0.5
            )  # !!! 0.5 comes from F.softplus !!!

            self.time_faaaa = nn.Parameter(torch.zeros(1, 1, n_head, head_size))
            self.time_aaaaa = nn.Parameter(torch.zeros(1, 1, dim_att))

            def ortho_init(x, scale):
                with torch.no_grad():
                    shape = x.shape
                    if len(shape) == 2:
                        gain = (
                            math.sqrt(shape[0] / shape[1]) if shape[0] > shape[1] else 1
                        )
                        nn.init.orthogonal_(x, gain=gain * scale)
                    elif len(shape) == 3:
                        gain = (
                            math.sqrt(shape[1] / shape[2]) if shape[1] > shape[2] else 1
                        )
                        for i in range(shape[0]):
                            nn.init.orthogonal_(x[i], gain=gain * scale)
                    else:
                        assert False
                    return x

            D_MIX_LORA = 32
            self.time_maa_w1 = nn.Parameter(torch.zeros(n_embd, D_MIX_LORA * 4))
            self.time_maa_w2 = nn.Parameter(
                ortho_init(torch.zeros(4, D_MIX_LORA, n_embd), 0.1)
            )

            D_DECAY_LORA = 64
            self.time_decay_w1 = nn.Parameter(torch.zeros(n_embd, D_DECAY_LORA))
            self.time_decay_w2 = nn.Parameter(
                ortho_init(torch.zeros(D_DECAY_LORA, dim_att), 0.1)
            )

            D_AAA_LORA = 16
            self.time_aaa_w1 = nn.Parameter(torch.zeros(n_embd, D_AAA_LORA))
            self.time_aaa_w2 = nn.Parameter(
                ortho_init(torch.zeros(D_AAA_LORA, dim_att), 0.1)
            )

            D_KKK_LORA = 16
            self.time_kkk_w1 = nn.Parameter(torch.zeros(n_embd, D_KKK_LORA))
            self.time_kkk_w2 = nn.Parameter(
                ortho_init(torch.zeros(D_KKK_LORA, dim_att), 0.1)
            )

            D_GATE_LORA = 128
            self.gate_w1 = nn.Parameter(
                ortho_init(torch.zeros(n_embd, D_GATE_LORA), 0.1)
            )
            self.gate_w2 = nn.Parameter(
                ortho_init(torch.zeros(D_GATE_LORA, dim_att), 0.1)
            )

            D_MA_LORA = 16
            self.ma_w1 = nn.Parameter(torch.zeros(n_embd, D_MA_LORA))
            self.ma_w2 = nn.Parameter(ortho_init(torch.zeros(D_MA_LORA, dim_att), 0.1))
            self.time_misc_a = nn.Parameter(torch.zeros(1, 1, n_embd))
            D_MK_LORA = 16
            self.mk_w1 = nn.Parameter(torch.zeros(n_embd, D_MK_LORA))
            self.mk_w2 = nn.Parameter(ortho_init(torch.zeros(D_MK_LORA, dim_att), 0.1))
            self.time_misc_k = nn.Parameter(torch.zeros(1, 1, n_embd))

            self.receptance = nn.Linear(n_embd, dim_att, bias=False)
            self.key = nn.Linear(n_embd, dim_att, bias=False)
            self.value = nn.Linear(n_embd, dim_att, bias=False)
            self.output = nn.Linear(dim_att, n_embd, bias=False)
            self.ln_x = nn.GroupNorm(n_head, dim_att)  # , eps=64e-5)

            self.receptance.weight.data.uniform_(
                -0.5 / (self.n_embd**0.5), 0.5 / (self.n_embd**0.5)
            )
            self.key.weight.data.uniform_(
                -0.05 / (self.n_embd**0.5), 0.05 / (self.n_embd**0.5)
            )
            self.value.weight.data.uniform_(
                -0.5 / (self.n_embd**0.5), 0.5 / (self.n_embd**0.5)
            )
            self.output.weight.data.zero_()

            if shift_mode == "q_shift_multihead":
                self.shift_func = q_shift_multihead
            elif shift_mode == "conv":
                self.shift_func = ShiftByConv(n_embd)
            elif shift_mode == "q_shift":
                self.shift_func = partial(groups_q_shift, K=scan_K)

    def jit_forward(self, x, txt=None, patch_resolution=None, mm_tokens=None):
        B, T, C = x.size()
        H = self.n_head

        #! shift will make the artifacts in corners
        if self.shift_mode == "q_shift_multihead":
            xx = self.shift_func(
                x,
                self.shift_pixel,
                patch_resolution=patch_resolution,
                with_cls_token=self.with_cls_token,
            )
        elif self.shift_mode == "q_shift":
            xx = self.shift_func(
                x, self.shift_pixel, H=patch_resolution[0], W=patch_resolution[1]
            )

        elif self.shift_mode == "conv":
            xx = self.shift_func(x, patch_resolution[0], patch_resolution[1])
        else:
            xx = x

        # fusion img_feature and llm_feature
        if txt is not None:
            if mm_tokens is not None:
                x = self.cat_special_tokens_to_seqs(x, txt, *mm_tokens)
                xx = self.cat_special_tokens_to_seqs(xx, txt, *mm_tokens)
                T = T + txt.size(1) + 4
            else:
                # print('spatial mix cat order:', self.img_txt_cat_order)
                # x = torch.cat([x, txt], dim=-1)
                # xx = torch.cat([xx, txt], dim=-1)
                x = self.cat_seqs_wo_special_tokens(x, txt)
                xx = self.cat_seqs_wo_special_tokens(xx, txt)
                T = T + txt.size(1)

        xxx = x + xx * self.time_maa_x
        xxx = torch.tanh(xxx @ self.time_maa_w1).view(B * T, 4, -1).transpose(0, 1)
        xxx = torch.bmm(xxx, self.time_maa_w2).view(4, B, T, -1)
        mrg, mwa, mk, mv = xxx.unbind(dim=0)

        xrg = x + xx * (self.time_maa_rg + mrg)
        xwa = x + xx * (self.time_maa_wa + mwa)
        xk = x + xx * (self.time_maa_k + mk)
        xv = x + xx * (self.time_maa_v + mv)

        r = self.receptance(xrg)
        w = (
            -F.softplus(
                -(
                    self.time_decay
                    + torch.tanh(xwa @ self.time_decay_w1) @ self.time_decay_w2
                )
            )
            - 0.5
        )
        k = self.key(xk)
        v = self.value(xv)
        g = torch.tanh(xrg @ self.gate_w1) @ self.gate_w2

        kk = k + torch.tanh(xk @ self.time_kkk_w1) @ self.time_kkk_w2
        kk = F.normalize(kk.view(B, T, H, -1), dim=-1, p=2.0).view(B, T, C)
        a = torch.sigmoid(self.time_aaaaa + (xwa @ self.time_aaa_w1) @ self.time_aaa_w2)

        ma = torch.sigmoid(self.time_misc_a + (xwa @ self.ma_w1) @ self.ma_w2)
        k = k * ma + k * a * (1 - ma)
        mk = torch.sigmoid(self.time_misc_k + (xk @ self.mk_w1) @ self.mk_w2)
        k = k * torch.clamp(w * mk, max=0).exp()

        x = RUN_CUDA_RWKV7g(r, w, k, v, -kk, (kk * a))
        x = self.ln_x(x.view(B * T, C)).view(B, T, C)

        x = x + (
            (r.view(B, T, H, -1) * k.view(B, T, H, -1) * self.time_faaaa).sum(
                dim=-1, keepdim=True
            )
            * v.view(B, T, H, -1)
        ).view(B, T, C)
        x = self.output(x * g)
        return x

    def cat_special_tokens_to_seqs(
        self,
        img: torch.Tensor,
        txt: torch.Tensor,
        img_token: torch.Tensor,
        txt_token: torch.Tensor,
    ):
        # add special tokens to sequences
        bs_txt, L_t, C_t = txt.shape
        bs_img, L_i, C_i = img.shape

        assert C_t == C_i, "modalities should have the same channels"

        if bs_txt == 1:
            txt = txt.repeat(bs_img, 1, 1)
            bs_txt = bs_img
        else:
            assert bs_txt == bs_img, "batch size of img and txt should be the same"

        soi = img_token[0].repeat(bs_img, 1, 1)
        eoi = img_token[1].repeat(bs_img, 1, 1)

        sot = txt_token[0].repeat(bs_txt, 1, 1)
        eot = txt_token[1].repeat(bs_txt, 1, 1)

        if self.img_txt_cat_order == 0:
            return torch.cat([soi, img, eoi, sot, txt, sot], dim=1)
        else:
            return torch.cat([sot, txt, eot, soi, img, eoi], dim=1)

    def cat_seqs_wo_special_tokens(self, img: torch.Tensor, txt: torch.Tensor):
        bs_txt, L_t, C_t = txt.shape
        bs_img, L_i, C_i = img.shape
        assert C_t == C_i, "modalities should have the same channels"

        if self.img_txt_cat_order == 0:
            return torch.cat([img, txt], dim=1)
        else:
            return torch.cat([txt, img], dim=1)

    def forward(self, x, txt=None, patch_resolution=None, mm_tokens=None):
        if self.with_cp and x.requires_grad:
            x = cp.checkpoint(
                self.jit_forward,
                x,
                txt,
                patch_resolution,
                mm_tokens,
                use_reentrant=False,
            )
        else:
            x = self.jit_forward(x, txt, patch_resolution, mm_tokens)
        return x


class VRWKV_SpatialMix_V6(nn.Module):
    def __init__(
        self,
        n_embd,
        n_head,
        n_layer,
        layer_id,
        shift_mode="none",
        shift_pixel=1,
        attn_bias=False,
        init_mode="fancy",
        key_norm=False,
        with_cls_token=False,
        with_cp=False,
        img_txt_cat_order=0,
        scan_K=1,
    ):
        super().__init__()
        self.layer_id = layer_id
        self.n_layer = n_layer
        self.n_embd = n_embd
        self.attn_sz = n_embd
        self.img_txt_cat_order = img_txt_cat_order
        self.scan_K = scan_K

        self.n_head = n_head
        self.head_size = self.attn_sz // self.n_head
        assert (
            self.head_size == HEAD_SIZE
        ), f"head size should be {HEAD_SIZE}, but got {self.head_size}"
        self.device = None
        self._init_weights(init_mode)
        self.with_cls_token = with_cls_token
        self.shift_pixel = shift_pixel
        self.shift_mode = shift_mode
        if shift_mode == "q_shift_multihead":
            self.shift_func = q_shift_multihead
        elif shift_mode == "conv":
            self.shift_func = ShiftByConv(self.n_embd)
        elif shift_mode == "q_shift":
            self.shift_func = partial(groups_q_shift, K=scan_K)

        self.key = nn.Linear(self.n_embd, self.attn_sz, bias=attn_bias)
        self.value = nn.Linear(self.n_embd, self.attn_sz, bias=attn_bias)
        self.receptance = nn.Linear(self.n_embd, self.attn_sz, bias=attn_bias)
        self.gate = nn.Linear(self.n_embd, self.attn_sz, bias=attn_bias)

        # NOTE: scanned image are from different directions, which are unsuitable from Linear, and in VMamba,
        #       the author recommand to use grouped Conv1d, but still the K*D channel of output feature
        #       is mixed by the directions. I choose to directly use einsum to isolate the K channel out of
        #       the D channel. So right here, I define the key, value, receptance, gate weight and ingore
        #       the bias by default.

        # _dim_in = self.n_embd // self.scan_K
        # _dim_out = self.attn_sz // self.scan_K
        # self.key_w = nn.Parameter(torch.empty(_dim_in, _dim_out))
        # self.value_w = nn.Parameter(torch.empty(_dim_in, _dim_out))
        # self.receptance_w = nn.Parameter(torch.empty(_dim_in, _dim_out))
        # self.gate_w = nn.Parameter(torch.empty(_dim_in, _dim_out))
        # # init
        # nn.init.xavier_normal_(self.key_w)
        # nn.init.xavier_normal_(self.value_w)
        # nn.init.xavier_normal_(self.receptance_w)
        # nn.init.xavier_normal_(self.gate_w)

        # if attn_bias:
        #     self.key_bias = nn.Parameter(torch.zeros(_dim_out))
        #     self.value_bias = nn.Parameter(torch.zeros(_dim_out))
        #     self.receptance_bias = nn.Parameter(torch.zeros(_dim_out))
        #     self.gate_bias = nn.Parameter(torch.zeros(_dim_out))

        if key_norm:
            self.key_norm = nn.LayerNorm(n_embd)
        else:
            self.key_norm = None
        self.output = nn.Linear(self.attn_sz, n_embd, bias=attn_bias)

        self.ln_x = nn.LayerNorm(self.attn_sz)
        self.with_cp = with_cp

    def _init_weights(self, init_mode):
        if init_mode == "fancy":
            with torch.no_grad():
                ratio_0_to_1 = self.layer_id / (self.n_layer - 1)  # 0 to 1
                ratio_1_to_almost0 = 1.0 - (self.layer_id / self.n_layer)  # 1 to ~0
                ddd = torch.ones(1, 1, self.n_embd)
                for i in range(self.n_embd):
                    ddd[0, 0, i] = i / self.n_embd

                # fancy time_mix
                self.time_maa_x = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))
                self.time_maa_w = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))
                self.time_maa_k = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))
                self.time_maa_v = nn.Parameter(
                    1.0 - (torch.pow(ddd, ratio_1_to_almost0) + 0.3 * ratio_0_to_1)
                )
                self.time_maa_r = nn.Parameter(
                    1.0 - torch.pow(ddd, 0.5 * ratio_1_to_almost0)
                )
                self.time_maa_g = nn.Parameter(
                    1.0 - torch.pow(ddd, 0.5 * ratio_1_to_almost0)
                )

                TIME_MIX_EXTRA_DIM = HEAD_SIZE  # generate TIME_MIX for w,k,v,r,g
                self.time_maa_w1 = nn.Parameter(
                    torch.zeros(self.n_embd, TIME_MIX_EXTRA_DIM * 5).uniform_(
                        -1e-4, 1e-4
                    )
                )
                self.time_maa_w2 = nn.Parameter(
                    torch.zeros(5, TIME_MIX_EXTRA_DIM, self.n_embd).uniform_(
                        -1e-4, 1e-4
                    )
                )

                # fancy time_decay
                decay_speed = torch.ones(self.attn_sz)
                for n in range(self.attn_sz):
                    decay_speed[n] = -6 + 5 * (n / (self.attn_sz - 1)) ** (
                        0.7 + 1.3 * ratio_0_to_1
                    )
                self.time_decay = nn.Parameter(decay_speed.reshape(1, 1, self.attn_sz))

                TIME_DECAY_EXTRA_DIM = TIME_DECAY_DIM
                self.time_decay_w1 = nn.Parameter(
                    torch.zeros(self.n_embd, TIME_DECAY_EXTRA_DIM).uniform_(-1e-4, 1e-4)
                )
                self.time_decay_w2 = nn.Parameter(
                    torch.zeros(TIME_DECAY_EXTRA_DIM, self.attn_sz).uniform_(
                        -1e-4, 1e-4
                    )
                )

                tmp = torch.zeros(self.attn_sz)
                for n in range(self.attn_sz):
                    zigzag = ((n + 1) % 3 - 1) * 0.1
                    tmp[n] = ratio_0_to_1 * (1 - (n / (self.attn_sz - 1))) + zigzag

                self.time_faaaa = nn.Parameter(tmp.reshape(self.n_head, self.head_size))
        else:
            raise NotImplementedError

    def isolate_scan_linear(self, x, weight):
        # x: [B, L, K, D]
        # w: [D, K]

        # isolate k first
        B, L, KD = x.shape
        x = x.view(B, L, self.scan_K, -1)

        return torch.einsum("blkd,dc->blkc", x, weight).view(B, L, -1)

    def cat_special_tokens_to_seqs(
        self,
        img: torch.Tensor,
        txt: torch.Tensor,
        img_token: torch.Tensor,
        txt_token: torch.Tensor,
    ):
        # add special tokens to sequences
        bs_txt, L_t, C_t = txt.shape
        bs_img, L_i, C_i = img.shape

        assert C_t == C_i, "modalities should have the same channels"

        if bs_txt == 1:
            txt = txt.repeat(bs_img, 1, 1)
            bs_txt = bs_img
        else:
            assert bs_txt == bs_img, "batch size of img and txt should be the same"

        soi = img_token[0].repeat(bs_img, 1, 1)
        eoi = img_token[1].repeat(bs_img, 1, 1)

        sot = txt_token[0].repeat(bs_txt, 1, 1)
        eot = txt_token[1].repeat(bs_txt, 1, 1)

        if self.img_txt_cat_order == 0:
            return torch.cat([soi, img, eoi, sot, txt, sot], dim=1)
        else:
            return torch.cat([sot, txt, eot, soi, img, eoi], dim=1)

    def cat_seqs_wo_special_tokens(self, img: torch.Tensor, txt: torch.Tensor):
        bs_txt, L_t, C_t = txt.shape
        bs_img, L_i, C_i = img.shape
        assert C_t == C_i, "modalities should have the same channels"

        if self.img_txt_cat_order == 0:
            return torch.cat([img, txt], dim=1)
        else:
            return torch.cat([txt, img], dim=1)

    def jit_func(self, x, txt=None, patch_resolution=None, mm_tokens=None):
        # Mix x with the previous timestep to produce xk, xv, xr
        B, T, C = x.size()

        #! shift will make the artifacts in corners
        if self.shift_mode == "q_shift_multihead":
            xx = self.shift_func(
                x,
                self.shift_pixel,
                patch_resolution=patch_resolution,
                with_cls_token=self.with_cls_token,
            )
        elif self.shift_mode == "q_shift":
            xx = self.shift_func(
                x, self.shift_pixel, H=patch_resolution[0], W=patch_resolution[1]
            )

        elif self.shift_mode == "conv":
            xx = self.shift_func(x, patch_resolution[0], patch_resolution[1])
        else:
            xx = x

        # fusion img_feature and llm_feature
        if txt is not None:
            if mm_tokens is not None:
                x = self.cat_special_tokens_to_seqs(x, txt, *mm_tokens)
                xx = self.cat_special_tokens_to_seqs(xx, txt, *mm_tokens)
                T = T + txt.size(1) + 4
            else:
                # print('spatial mix cat order:', self.img_txt_cat_order)
                # x = torch.cat([x, txt], dim=-1)
                # xx = torch.cat([xx, txt], dim=-1)
                x = self.cat_seqs_wo_special_tokens(x, txt)
                xx = self.cat_seqs_wo_special_tokens(xx, txt)
                T = T + txt.size(1)

        xxx = x * self.time_maa_x + xx * (1 - self.time_maa_x)
        # xxx = x + xx * self.time_maa_x

        # [B, T, C] @ [C, 5*HEAD_SIZE] -> [B, T, 5*HEAD_SIZE]
        xxx = torch.tanh(xxx @ self.time_maa_w1).view(B * T, 5, -1).transpose(0, 1)
        # [5, B*T, TIME_MIX_EXTRA_DIM]
        xxx = torch.bmm(xxx, self.time_maa_w2).view(5, B, T, -1)
        # [5, B, T, C]
        mw, mk, mv, mr, mg = xxx.unbind(dim=0)

        xw = x + xx * (self.time_maa_w + mw)
        xk = x + xx * (self.time_maa_k + mk)
        xv = x + xx * (self.time_maa_v + mv)
        xr = x + xx * (self.time_maa_r + mr)
        xg = x + xx * (self.time_maa_g + mg)

        r = self.receptance(xr)
        k = self.key(xk)
        v = self.value(xv)
        g = F.silu(self.gate(xg))

        # r = self.isolate_scan_linear(xr, self.receptance_w)
        # k = self.isolate_scan_linear(xk, self.key_w)
        # v = self.isolate_scan_linear(xv, self.value_w)
        # g = F.silu(self.isolate_scan_linear(xg, self.gate_w))

        ww = torch.tanh(xw @ self.time_decay_w1) @ self.time_decay_w2
        # [B, T, C]
        w = self.time_decay + ww

        return r, k, v, g, w, T

    def jit_func_2(self, x, g):
        B, T, C = x.size()

        x = self.ln_x(x).view(B, T, C)
        x = self.output(x * g)
        return x

    def forward(
        self,
        x,
        txt=None,
        patch_resolution=None,
        mm_tokens: "tuple[torch.Tensor] | None" = None,
    ):
        def _inner_forward(x, txt, patch_resolution):
            B, T, C = x.size()
            self.device = x.device

            r, k, v, g, w, T = self.jit_func(x, txt, patch_resolution, mm_tokens)
            x = RUN_CUDA_RWKV6(B, T, C, self.n_head, r, k, v, w, u=self.time_faaaa)
            if self.key_norm is not None:
                x = self.key_norm(x)
            return self.jit_func_2(x, g)

        if self.with_cp and x.requires_grad:
            x = cp.checkpoint(
                _inner_forward, x, txt, patch_resolution, use_reentrant=False
            )
        else:
            x = _inner_forward(x, txt, patch_resolution)
        return x


class VRWKV_ChannelMix(nn.Module):
    def __init__(
        self,
        n_embd,
        n_head,
        n_layer,
        layer_id,
        shift_mode="none",
        shift_pixel=1,
        hidden_rate=4,
        ffn_bias=False,
        init_mode="fancy",
        key_norm=False,
        with_cls_token=False,
        with_cp=False,
        scan_K=1,
        img_txt_cat_order=0,
    ):
        super().__init__()
        self.layer_id = layer_id
        self.n_layer = n_layer
        self.n_embd = n_embd
        self.attn_sz = n_embd
        self.n_head = n_head
        self.head_size = self.attn_sz // self.n_head
        self.img_txt_cat_order = img_txt_cat_order
        assert self.head_size == HEAD_SIZE
        self.with_cp = with_cp
        self._init_weights(init_mode)
        self.with_cls_token = with_cls_token
        self.shift_pixel = shift_pixel
        self.shift_mode = shift_mode
        if shift_mode == "q_shift_multihead":
            self.shift_func = q_shift_multihead
        elif shift_mode == "q_shift":
            self.shift_func = partial(groups_q_shift, K=scan_K)
        elif shift_mode == "conv":
            self.shift_func = ShiftByConv(self.n_embd)

        hidden_sz = hidden_rate * n_embd
        self.key = nn.Linear(n_embd, hidden_sz, bias=False)
        if key_norm:
            self.key_norm = nn.LayerNorm(hidden_sz)
        else:
            self.key_norm = None
        self.receptance = nn.Linear(n_embd, n_embd, bias=ffn_bias)
        self.value = nn.Linear(hidden_sz, n_embd, bias=ffn_bias)

    def _init_weights(self, init_mode):
        if init_mode == "fancy":
            with torch.no_grad():  # fancy init of time_mix
                ratio_1_to_almost0 = 1.0 - (self.layer_id / self.n_layer)  # 1 to ~0
                x = torch.ones(1, 1, self.n_embd)
                for i in range(self.n_embd):
                    x[0, 0, i] = i / self.n_embd
                self.spatial_mix_k = nn.Parameter(torch.pow(x, ratio_1_to_almost0))
                self.spatial_mix_r = nn.Parameter(torch.pow(x, ratio_1_to_almost0))
        else:
            raise NotImplementedError

    ########################################################## Channel mixing multi-modal version #########################################################
    # NOTE: channel mixing cat same as spatial mixing
    # but we do not use this multi-modal version, the mix concatenation in a simple MLP

    def cat_special_tokens_to_seqs(
        self,
        img: torch.Tensor,
        txt: torch.Tensor,
        img_token: torch.Tensor,
        txt_token: torch.Tensor,
    ):
        # add special tokens to sequences
        bs_txt, L_t, C_t = txt.shape
        bs_img, L_i, C_i = img.shape

        assert C_t == C_i, "modalities should have the same channels"

        if bs_txt == 1:
            txt = txt.repeat(bs_img, 1, 1)
            bs_txt = bs_img
        else:
            assert bs_txt == bs_img, "batch size of img and txt should be the same"

        soi = img_token[0].repeat(bs_img, 1, 1)
        eoi = img_token[1].repeat(bs_img, 1, 1)

        sot = txt_token[0].repeat(bs_txt, 1, 1)
        eot = txt_token[1].repeat(bs_txt, 1, 1)

        if self.img_txt_cat_order == 0:
            return torch.cat([soi, img, eoi, sot, txt, sot], dim=1)
        else:
            return torch.cat([sot, txt, eot, soi, img, eoi], dim=1)

    def cat_seqs_wo_special_tokens(self, img: torch.Tensor, txt: torch.Tensor):
        bs_txt, L_t, C_t = txt.shape
        bs_img, L_i, C_i = img.shape
        assert C_t == C_i, "modalities should have the same channels"

        if self.img_txt_cat_order == 0:
            return torch.cat([img, txt], dim=1)
        else:
            return torch.cat([txt, img], dim=1)

    #########################################################################################################################################################

    def forward(
        self,
        x,
        txt=None,
        patch_resolution=None,
        mm_tokens: "tuple[torch.Tensor] | None" = None,
    ):
        def _inner_forward(x, txt, patch_resolution):
            B, C, T = x.size()

            if self.shift_mode == "q_shift_multihead":
                xx = self.shift_func(
                    x,
                    self.shift_pixel,
                    patch_resolution=patch_resolution,
                    with_cls_token=self.with_cls_token,
                )
            elif self.shift_mode == "q_shift":
                xx = self.shift_func(
                    x, self.shift_pixel, H=patch_resolution[0], W=patch_resolution[1]
                )
            elif self.shift_mode == "conv":
                xx = self.shift_func(x, patch_resolution[0], patch_resolution[1])
            else:
                xx = x

            if txt is not None:
                if mm_tokens is not None:
                    x = self.cat_special_tokens_to_seqs(x, txt, *mm_tokens)
                    xx = self.cat_special_tokens_to_seqs(xx, txt, *mm_tokens)
                    T = T + txt.size(-1) + 4
                else:
                    # x = torch.cat([x, txt], dim=-1)
                    # xx = torch.cat([xx, txt], dim=-1)
                    # print('channel mix cat order:', self.img_txt_cat_order)
                    x = self.cat_seqs_wo_special_tokens(x, txt)
                    xx = self.cat_seqs_wo_special_tokens(xx, txt)
                    T = T + txt.size(-1)

            xk = x * self.spatial_mix_k + xx * (1 - self.spatial_mix_k)
            xr = x * self.spatial_mix_r + xx * (1 - self.spatial_mix_r)

            k = self.key(xk)
            k = torch.square(torch.relu(k))
            if self.key_norm is not None:
                k = self.key_norm(k)
            kv = self.value(k)
            x = torch.sigmoid(self.receptance(xr)) * kv
            return x

        if self.with_cp and x.requires_grad:
            x = cp.checkpoint(
                _inner_forward, x, txt, patch_resolution, use_reentrant=False
            )
        else:
            x = _inner_forward(x, txt, patch_resolution)
        return x


#################################################################################################################

########################################## Omini-shift RWKV (simplify version) ##########################################


class OmniShift(nn.Module):
    def __init__(self, dim):
        super(OmniShift, self).__init__()
        # Define the layers for training
        self.conv1x1 = nn.Conv2d(
            in_channels=dim, out_channels=dim, kernel_size=1, groups=dim, bias=False
        )
        self.conv3x3 = nn.Conv2d(
            in_channels=dim,
            out_channels=dim,
            kernel_size=3,
            padding=1,
            groups=dim,
            bias=False,
        )
        self.conv5x5 = nn.Conv2d(
            in_channels=dim,
            out_channels=dim,
            kernel_size=5,
            padding=2,
            groups=dim,
            bias=False,
        )
        self.alpha = nn.Parameter(torch.randn(4), requires_grad=True)

        # Define the layers for testing
        # self.conv5x5_reparam = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=5, padding=2, groups=dim, bias = False)
        self.repram_flag = False

    def forward_train(self, x):
        out1x1 = self.conv1x1(x)
        out3x3 = self.conv3x3(x)
        out5x5 = self.conv5x5(x)

        out = (
            self.alpha[0] * x
            + self.alpha[1] * out1x1
            + self.alpha[2] * out3x3
            + self.alpha[3] * out5x5
        )
        return out

    def reparam_5x5(self):
        # Combine the parameters of conv1x1, conv3x3, and conv5x5 to form a single 5x5 depth-wise convolution
        padded_weight_1x1 = F.pad(self.conv1x1.weight, (2, 2, 2, 2))
        padded_weight_3x3 = F.pad(self.conv3x3.weight, (1, 1, 1, 1))

        identity_weight = F.pad(torch.ones_like(self.conv1x1.weight), (2, 2, 2, 2))
        combined_weight = (
            self.alpha[0] * identity_weight
            + self.alpha[1] * padded_weight_1x1
            + self.alpha[2] * padded_weight_3x3
            + self.alpha[3] * self.conv5x5.weight
        )
        device = self.conv5x5_reparam.weight.device
        combined_weight = combined_weight.to(device)
        # self.conv5x5_reparam_weight = nn.Parameter(combined_weight)
        self.conv5x5_reparam_weight = combined_weight

    def conv5x5_reparam(self, x):
        assert self.repram_flag == False, "repram flag must be False"
        out = F.conv2d(x, self.conv5x5_reparam_weight, bias=None, stride=1, padding=2)
        return out

    def forward(self, x):
        # if self.training:
        #     self.repram_flag = True
        #     out = self.forward_train(x)
        # elif self.training == False and self.repram_flag == True:
        #     self.reparam_5x5()
        #     self.repram_flag = False
        #     out = self.conv5x5_reparam(x)
        # elif self.training == False and self.repram_flag == False:
        #     out = self.conv5x5_reparam(x)

        out = self.forward_train(x)

        return out


class VRWKV_SpatialMix_wkv5(nn.Module):
    def __init__(
        self,
        n_embd,
        n_head,
        n_layer,
        layer_id,
        shift_mode="none",
        shift_pixel=1,
        attn_bias=False,
        init_mode="fancy",
        key_norm=True,
        with_cls_token=False,
        with_cp=False,
        img_txt_cat_order=0,
        scan_K=1,
        head_div=1,
        decay_speed="slow",
    ):
        super().__init__()
        self.layer_id = layer_id
        self.n_layer = n_layer
        self.n_embd = n_embd
        attn_sz = n_embd
        self.with_cp = with_cp
        self.img_txt_cat_order = img_txt_cat_order
        self.scan_K = scan_K
        self.n_head = n_head
        head_size = n_embd // n_head
        assert head_size * n_head == n_embd, "n_embd should be divisible by n_head"
        self.head_size = head_size
        self.head_size_divisor = head_div

        # shift
        # self.omni_shift = OmniShift(dim=n_embd)

        # key, value, receptance
        self.key = nn.Linear(n_embd, attn_sz, bias=False)
        self.value = nn.Linear(n_embd, attn_sz, bias=False)
        self.receptance = nn.Linear(n_embd, attn_sz, bias=False)
        if key_norm:
            self.key_norm = nn.LayerNorm(n_embd)
        else:
            self.key_norm = partial(F.normalize, dim=-1)
        self.output = nn.Linear(attn_sz, n_embd, bias=False)
        self.gate = nn.Linear(n_embd, attn_sz, bias=False)
        # self.ln_x = nn.GroupNorm(n_head, attn_sz)
        self.ln_x = nn.LayerNorm(attn_sz)

        # spatial decay and spatial first
        with torch.no_grad():
            ratio_0_to_1 = layer_id / (n_layer - 1)  # 0 to 1
            ratio_1_to_almost0 = 1.0 - (layer_id / n_layer)  # 1 to ~0
            ddd = torch.ones(1, 1, n_embd)
            for i in range(n_embd):
                ddd[0, 0, i] = i / n_embd

            # fancy time_mix
            # self.time_mix_k = nn.Parameter(torch.pow(ddd, ratio_1_to_almost0))
            # self.time_mix_v = nn.Parameter(torch.pow(ddd, ratio_1_to_almost0) + 0.3 * ratio_0_to_1)
            # self.time_mix_r = nn.Parameter(torch.pow(ddd, 0.5 * ratio_1_to_almost0))
            # self.time_mix_g = nn.Parameter(torch.pow(ddd, 0.5 * ratio_1_to_almost0))

            # fancy time_decay
            decay_speed = torch.ones(attn_sz)
            for n in range(attn_sz):
                if decay_speed == "slow":
                    decay_speed[n] = -6 + 5 * (n / (attn_sz - 1)) ** (
                        0.7 + 1.3 * ratio_0_to_1
                    )
                else:
                    decay_speed[n] = -0.3 + 1 * (n / (attn_sz - 1)) ** (
                        0.1 + 1.3 * ratio_0_to_1
                    )

            self.time_decay = nn.Parameter(
                decay_speed.reshape(self.n_head, self.head_size)
            )
            # print(layer_id, self.time_decay.flatten()[:3].cpu().numpy(), '...', self.time_decay.flatten()[-3:].cpu().numpy())

            tmp = torch.zeros(attn_sz)
            for n in range(attn_sz):
                zigzag = ((n + 1) % 3 - 1) * 0.1
                tmp[n] = ratio_0_to_1 * (1 - (n / (attn_sz - 1))) + zigzag

            self.time_faaaa = nn.Parameter(tmp.reshape(self.n_head, self.head_size))

    def cat_special_tokens_to_seqs(
        self,
        img: torch.Tensor,
        txt: torch.Tensor,
        img_token: torch.Tensor,
        txt_token: torch.Tensor,
    ):
        # add special tokens to sequences
        bs_txt, L_t, C_t = txt.shape
        bs_img, L_i, C_i = img.shape

        assert C_t == C_i, "modalities should have the same channels"

        if bs_txt == 1:
            txt = txt.repeat(bs_img, 1, 1)
            bs_txt = bs_img
        else:
            assert bs_txt == bs_img, "batch size of img and txt should be the same"

        soi = img_token[0].repeat(bs_img, 1, 1)
        eoi = img_token[1].repeat(bs_img, 1, 1)

        sot = txt_token[0].repeat(bs_txt, 1, 1)
        eot = txt_token[1].repeat(bs_txt, 1, 1)

        if self.img_txt_cat_order == 0:
            return torch.cat([soi, img, eoi, sot, txt, sot], dim=1)
        else:
            return torch.cat([sot, txt, eot, soi, img, eoi], dim=1)

    def cat_seqs_wo_special_tokens(self, img: torch.Tensor, txt: torch.Tensor):
        bs_txt, L_t, C_t = txt.shape
        bs_img, L_i, C_i = img.shape
        assert C_t == C_i, "modalities should have the same channels"

        if self.img_txt_cat_order == 0:
            return torch.cat([img, txt], dim=1)
        else:
            return torch.cat([txt, img], dim=1)

    # def shift_scanned_x(self, x, h, w):
    #     assert self.scan_K in [0, 2], "scan_K should be 0 or 2"
    #     if self.scan_K == 2:
    #         xx = rearrange(x, '(b k) l c -> b k c l', h=h, w=w, k=self.scan_K)
    #         # same
    #         xx_0 = rearrange(xx[:, 0], 'b c (h w) -> b c h w', h=h, w=w)
    #         xx_0 = self.omni_shift(xx_0)
    #         xx_0 = rearrange(xx_0, 'b c h w -> b c (h w)')
    #         # transposed
    #         xx_1 = rearrange(xx[:, 1], 'b c (w h) -> b c h w', h=h, w=w)
    #         xx_1 = self.omni_shift(xx_1)
    #         xx_1 = rearrange(xx_1, 'b c h w -> b c (h w)')

    #         xx = torch.stack([xx_0, xx_1], dim=1)

    #         xx = rearrange(xx, 'b k c h w -> b c h w')
    #         return xx

    def jit_func(
        self,
        x,
        txt=None,
        resolution=None,
        mm_tokens: "tuple[torch.Tensor] | None" = None,
    ):
        # Mix x with the previous timestep to produce xk, xv, xr
        B, T, C = x.size()
        h, w = resolution

        # xx = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
        # xx = self.omni_shift(xx)
        # xx = rearrange(xx, 'b c h w -> b (h w) c')

        # xx = x

        # cat txt and special tokens
        if txt is not None:
            if mm_tokens is not None:
                x = self.cat_special_tokens_to_seqs(x, txt, *mm_tokens)
                # xx = self.cat_special_tokens_to_seqs(xx, txt, *mm_tokens)
                T = T + txt.size(1) + 4
            else:
                x = self.cat_seqs_wo_special_tokens(x, txt)
                # xx = self.cat_seqs_wo_special_tokens(xx, txt)
                T = T + txt.size(1)

        # time decay
        # xk = x * self.time_mix_k + xx * (1 - self.time_mix_k)
        # xv = x * self.time_mix_v + xx * (1 - self.time_mix_v)
        # xr = x * self.time_mix_r + xx * (1 - self.time_mix_r)
        # xg = x * self.time_mix_g + xx * (1 - self.time_mix_g)

        k = self.key(x)
        v = self.value(x)
        r = self.receptance(x)
        # sr = torch.sigmoid(r)
        g = F.silu(self.gate(x))

        return r, k, v, g, T

    def jit_func_2(self, x, g):
        # B, T, C = x.size()
        # x = x.reshape(B * T, C)
        # x = self.ln_x(x / self.head_size_divisor).reshape(B, T, C)

        x = self.ln_x(x)
        x = self.output(x * g)
        return x

    def forward(
        self,
        x,
        txt=None,
        patch_resolution=None,
        mm_tokens: "tuple[torch.Tensor] | None" = None,
    ):
        def _inner_forward(x, txt, patch_resolution, mm_tokens):
            B, T, C = x.size()
            # sr, k, v, T = self.jit_func(x, txt, patch_resolution, mm_tokens)
            r, k, v, g, T = self.jit_func(x, txt, patch_resolution, mm_tokens)
            # x = RUN_CUDA_RWKV5(B, T, C, self.spatial_decay / T, self.spatial_first / T, k, v)
            x = RUN_CUDA_RWKV5_2(
                B, T, C, self.n_head, r, k, v, w=self.time_decay, u=self.time_faaaa
            )
            # print(f'Spatial mix - x max: {x.abs().max()}, x norm: {x.norm()}')
            x = self.key_norm(x)
            # x = sr * x
            # x = self.output(x)
            x = self.jit_func_2(x, g)

            return x

        if self.with_cp and x.requires_grad:
            x = cp.checkpoint(
                _inner_forward, x, txt, patch_resolution, mm_tokens, use_reentrant=False
            )
        else:
            x = _inner_forward(x, txt, patch_resolution, mm_tokens)
        return x


class VRWKV_ChannelMix_wkv5(nn.Module):
    def __init__(
        self,
        n_embd,
        n_head,
        n_layer,
        layer_id,
        shift_mode="none",
        shift_pixel=1,
        hidden_rate=4,
        ffn_bias=False,
        init_mode="fancy",
        key_norm=False,
        with_cls_token=False,
        with_cp=False,
        scan_K=1,
        img_txt_cat_order=0,
    ):
        super().__init__()
        self.layer_id = layer_id
        self.n_layer = n_layer
        self.n_embd = n_embd
        self.n_head = n_head
        self.hidden_rate = hidden_rate
        self.ffn_bias = ffn_bias
        self.init_mode = init_mode
        self.key_norm = key_norm
        self.with_cp = with_cp
        self.scan_K = scan_K
        self.img_txt_cat_order = img_txt_cat_order

        hidden_sz = int(hidden_rate * n_embd)
        self.key = nn.Linear(n_embd, hidden_sz, bias=False)

        # self.omni_shift = OmniShift(dim=n_embd)

        if key_norm:
            self.key_norm = nn.LayerNorm(hidden_sz)
        else:
            # self.key_norm = partial(F.normalize, dim=-1)
            self.key_norm = nn.Identity()

        self.receptance = nn.Linear(n_embd, n_embd, bias=False)
        self.value = nn.Linear(hidden_sz, n_embd, bias=False)

        # time decay
        # with torch.no_grad():  # fancy init of time_mix
        #     ratio_1_to_almost0 = 1.0 - (layer_id / n_layer)  # 1 to ~0
        #     ddd = torch.ones(1, 1, n_embd)
        #     for i in range(n_embd):
        #         ddd[0, 0, i] = i / n_embd
        #     self.time_mix_k = nn.Parameter(torch.pow(ddd, ratio_1_to_almost0))
        #     self.time_mix_r = nn.Parameter(torch.pow(ddd, ratio_1_to_almost0))

    def cat_special_tokens_to_seqs(
        self,
        img: torch.Tensor,
        txt: torch.Tensor,
        img_token: torch.Tensor,
        txt_token: torch.Tensor,
    ):
        # add special tokens to sequences
        bs_txt, L_t, C_t = txt.shape
        bs_img, L_i, C_i = img.shape

        assert C_t == C_i, "modalities should have the same channels"

        if bs_txt == 1:
            txt = txt.repeat(bs_img, 1, 1)
            bs_txt = bs_img
        else:
            assert bs_txt == bs_img, "batch size of img and txt should be the same"

        soi = img_token[0].repeat(bs_img, 1, 1)
        eoi = img_token[1].repeat(bs_img, 1, 1)

        sot = txt_token[0].repeat(bs_txt, 1, 1)
        eot = txt_token[1].repeat(bs_txt, 1, 1)

        if self.img_txt_cat_order == 0:
            return torch.cat([soi, img, eoi, sot, txt, sot], dim=1)
        else:
            return torch.cat([sot, txt, eot, soi, img, eoi], dim=1)

    def cat_seqs_wo_special_tokens(self, img: torch.Tensor, txt: torch.Tensor):
        bs_txt, L_t, C_t = txt.shape
        bs_img, L_i, C_i = img.shape
        assert C_t == C_i, "modalities should have the same channels"

        if self.img_txt_cat_order == 0:
            return torch.cat([img, txt], dim=1)
        else:
            return torch.cat([txt, img], dim=1)

    def forward(
        self,
        x,
        txt=None,
        patch_resolution=None,
        mm_tokens: "tuple[torch.Tensor] | None" = None,
    ):
        def _inner_forward(x, txt, patch_resolution, mm_tokens):
            B, T, C = x.size()
            h, w = patch_resolution

            # xx = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
            # xx = self.omni_shift(xx)
            # xx = rearrange(xx, 'b c h w -> b (h w) c')
            # xx = x

            if txt is not None:
                if mm_tokens is not None:
                    x = self.cat_special_tokens_to_seqs(x, txt, *mm_tokens)
                    # xx = self.cat_special_tokens_to_seqs(xx, txt, *mm_tokens)
                    T = T + txt.size(-1) + 4
                else:
                    x = self.cat_seqs_wo_special_tokens(x, txt)
                    # xx = self.cat_seqs_wo_special_tokens(xx, txt)
                    T = T + txt.size(-1)

            # xk = x * self.time_mix_k + xx * (1 - self.time_mix_k)
            # xr = x * self.time_mix_r + xx * (1 - self.time_mix_r)

            k = self.key(x)
            k = torch.square(torch.relu(k))
            # print(f'Channel mix - k max: {k.abs().max()}, k norm: {k.norm()}')
            k = self.key_norm(k)
            kv = self.value(k)

            x = torch.sigmoid(self.receptance(x)) * kv

            return x

        if self.with_cp and x.requires_grad:
            x = cp.checkpoint(
                _inner_forward, x, txt, patch_resolution, mm_tokens, use_reentrant=False
            )
        else:
            x = _inner_forward(x, txt, patch_resolution, mm_tokens)

        return x


######################################################## RWKV FLOPs ###############################################


def vrwkv6_flops(inp, outp):
    sizes = inp[0].type().sizes()
    B, T, C = sizes
    head_size = C // 32  # nhead: 32
    dim = C
    n = T

    TIME_MIX_EXTRA_DIM = 32
    TIME_DECAY_EXTRA_DIM = 64  # TIME_DECAY_DIM

    addi_flops = 0
    addi_flops += (
        n
        * dim
        * (TIME_MIX_EXTRA_DIM * 10 + TIME_DECAY_EXTRA_DIM * 2 + 7 * head_size + 17)
    )
    addi_flops += n * (TIME_MIX_EXTRA_DIM * 5 + TIME_DECAY_EXTRA_DIM)
    return addi_flops * B


class Block(nn.Module):
    def __init__(
        self,
        n_embd,
        n_head,
        n_layer,
        layer_id,
        shift_mode="q_shift_multihead",
        shift_pixel=1,
        drop_path=0.0,
        hidden_rate=4,
        init_mode="fancy",
        init_values=None,
        post_norm=False,
        key_norm=False,
        with_cls_token=False,
        with_cp=False,
    ):
        super().__init__()
        self.layer_id = layer_id
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        if self.layer_id == 0:
            self.ln0 = nn.LayerNorm(n_embd)

        self.att = VRWKV_SpatialMix_V6(
            n_embd,
            n_head,
            n_layer,
            layer_id,
            shift_mode,
            shift_pixel,
            init_mode,
            key_norm=key_norm,
            with_cls_token=with_cls_token,
        )

        self.ffn = VRWKV_ChannelMix(
            n_embd,
            n_head,
            n_layer,
            layer_id,
            shift_mode,
            shift_pixel,
            hidden_rate,
            init_mode,
            key_norm=key_norm,
            with_cls_token=with_cls_token,
        )
        self.layer_scale = init_values is not None
        self.post_norm = post_norm
        if self.layer_scale:
            self.gamma1 = nn.Parameter(
                init_values * torch.ones((n_embd)), requires_grad=True
            )
            self.gamma2 = nn.Parameter(
                init_values * torch.ones((n_embd)), requires_grad=True
            )
        self.with_cp = with_cp

    def forward(self, x, patch_resolution=None):
        def _inner_forward(x):
            if self.layer_id == 0:
                x = self.ln0(x)
            if self.post_norm:
                if self.layer_scale:
                    x = x + self.drop_path(
                        self.gamma1 * self.ln1(self.att(x, patch_resolution))
                    )
                    x = x + self.drop_path(
                        self.gamma2 * self.ln2(self.ffn(x, patch_resolution))
                    )
                else:
                    x = x + self.drop_path(self.ln1(self.att(x, patch_resolution)))
                    x = x + self.drop_path(self.ln2(self.ffn(x, patch_resolution)))
            else:
                if self.layer_scale:
                    x = x + self.drop_path(
                        self.gamma1 * self.att(self.ln1(x), patch_resolution)
                    )
                    x = x + self.drop_path(
                        self.gamma2 * self.ffn(self.ln2(x), patch_resolution)
                    )
                else:
                    x = x + self.drop_path(self.att(self.ln1(x), patch_resolution))
                    x = x + self.drop_path(self.ffn(self.ln2(x), patch_resolution))
            return x

        if self.with_cp and x.requires_grad:
            x = cp.checkpoint(_inner_forward, x)
        else:
            x = _inner_forward(x)
        return x


if __name__ == "__main__":
    # block = RWKV7(
    #     32, 16, 8, 1, with_cp=False,
    # ).cuda()

    # x = torch.randn(1, 512, 32).cuda()
    # print(block(x))

    spatial_mix = VRWKV_SpatialMix_wkv5(
        32,
        1,
        2,
        1,
    ).cuda()

    # channel_mix = VRWKV_ChannelMix_wkv5(
    #     32, 32, 2, 1
    # ).cuda()

    # spatial_mix = VRWKV_SpatialMix_V6(
    #     32, 1, 2, 1,
    # ).cuda()

    # spatial_mix = VRWKV_SpatialMix_V7(
    #     32, 1, 2, 1,
    # ).cuda()

    x = torch.randn(1, 256 * 256, 32).cuda()
    a = spatial_mix(x, None, (256, 256))
    print(a.shape)
    a.sum().backward()
    print(a.shape)
