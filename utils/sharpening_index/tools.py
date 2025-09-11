# -*- coding: utf-8 -*-
"""
Copyright (c) 2020 Vivone Gemine.
All rights reserved. This work should only be used for nonprofit purposes.

@author: Vivone Gemine (website: https://sites.google.com/site/vivonegemine/home )

Adapted versions of the functions used in:
    Python Code on GitHub: https://github.com/sergiovitale/pansharpening-cnn-python-version
    Copyright (c) 2018 Image Processing Research Group of University Federico II of Naples ('GRIP-UNINA').
"""

import numpy as np


def fir_filter_wind(Hd, w):
    hd = np.rot90(np.fft.fftshift(np.rot90(Hd, 2)), 2)
    h = np.fft.fftshift(np.fft.ifft2(hd))
    h = np.rot90(h, 2)
    h = h * w
    # h=h/np.sum(h)

    return h


def gaussian2d(N, std):
    t = np.arange(-(N - 1) / 2, (N + 1) / 2)
    # t=np.arange(-(N-1)/2,(N+2)/2)
    t1, t2 = np.meshgrid(t, t)
    std = np.double(std)
    w = np.exp(-0.5 * (t1 / std) ** 2) * np.exp(-0.5 * (t2 / std) ** 2)
    return w


def kaiser2d(N, beta):
    t = np.arange(-(N - 1) / 2, (N + 1) / 2) / np.double(N - 1)
    # t=np.arange(-(N-1)/2,(N+2)/2)/np.double(N-1)
    t1, t2 = np.meshgrid(t, t)
    t12 = np.sqrt(t1 * t1 + t2 * t2)
    w1 = np.kaiser(N, beta)
    w = np.interp(t12, t, w1)
    w[t12 > t[-1]] = 0
    w[t12 < t[0]] = 0

    return w


# * ==========================================================
# * Torch version

import torch
import torch.fft
import numpy as np


def fir_filter_wind_torch(Hd, w):
    """PyTorch implementation of FIR filter windowing"""
    # 确保输入是torch.Tensor
    if not isinstance(Hd, torch.Tensor):
        Hd = torch.tensor(Hd, dtype=torch.complex64)
    if not isinstance(w, torch.Tensor):
        w = torch.tensor(w, dtype=torch.float32)

    # 执行与NumPy版本相同的操作
    hd = torch.rot90(torch.fft.fftshift(torch.rot90(Hd, 2, [0, 1])), 2, [0, 1])
    h = torch.fft.fftshift(torch.fft.ifft2(hd).real)
    h = torch.rot90(h, 2, [0, 1])
    h = h * w

    return h


def gaussian2d_torch(N, std):
    """PyTorch implementation of 2D Gaussian window"""
    # 创建网格
    t = torch.arange(-(N - 1) / 2, (N + 1) / 2, dtype=torch.float32)
    t1, t2 = torch.meshgrid(t, t, indexing="ij")
    std = torch.tensor(std, dtype=torch.float32)

    # 计算高斯窗
    w = torch.exp(-0.5 * (t1 / std) ** 2) * torch.exp(-0.5 * (t2 / std) ** 2)
    return w


def kaiser2d_torch(N, beta):
    """PyTorch implementation of 2D Kaiser window"""
    # k_np = kaiser2d(N, beta)
    t = torch.arange(-(N - 1) / 2, (N + 1) / 2, dtype=torch.float32) / (N - 1)
    t1, t2 = torch.meshgrid(t, t)
    t12 = torch.sqrt(t1 * t1 + t2 * t2)

    w1 = torch.kaiser_window(
        N,
        periodic=False,
        beta=beta,
    )
    w = interp_torch(t12, t, w1)
    w[t12 > t[-1]] = 0
    w[t12 < t[0]] = 0

    return w


def interp_torch(x: torch.Tensor, xp: torch.Tensor, fp: torch.Tensor) -> torch.Tensor:
    x = x.float()
    xp = xp.float()
    fp = fp.float()

    indices = torch.searchsorted(xp, x)
    indices = torch.clamp(indices, 1, len(xp) - 1)

    x0 = xp[indices - 1]
    x1 = xp[indices]
    y0 = fp[indices - 1]
    y1 = fp[indices]

    result = y0 + (y1 - y0) * ((x - x0) / (x1 - x0))
    return result


if __name__ == "__main__":
    # k = fir_filter_wind(gaussian2d(5, 0.5), kaiser2d(5, 0.5))
    # k2 = fir_filter_wind_torch(
    #     gaussian2d_torch(5, 0.5), kaiser2d_torch(5, 0.5)
    # )

    k = kaiser2d(18, 12)
    k2 = kaiser2d_torch(18, 12)

    print("Kaiser2D Output Shape:", k.shape)
    print("Kaiser2D Torch Output Shape:", k2.shape)
    print((torch.tensor(k) - k2).abs().max())
