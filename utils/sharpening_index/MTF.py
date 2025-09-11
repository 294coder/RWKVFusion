# -*- coding: utf-8 -*-
"""
Copyright (c) 2020 Vivone Gemine.
All rights reserved. This work should only be used for nonprofit purposes.

@author: Vivone Gemine (website: https://sites.google.com/site/vivonegemine/home )
"""

"""
 Description:
           MTF filters the image I_MS using a Gaussin filter matched with the Modulation Transfer Function (MTF) of the MultiSpectral (MS) sensor.

 Interface:
           I_Filtered = MTF(I_MS,sensor,ratio)

 Inputs:
           I_MS:           MS image;
           sensor:         String for type of sensor (e.g. 'WV2', 'IKONOS');
           ratio:          Scale ratio between MS and PAN.

 Outputs:
           I_Filtered:     Output filtered MS image.

 Notes:
     The bottleneck of this function is the function scipy.filters.correlate that gets the same results as in the MATLAB toolbox
     but it is very slow with respect to fftconvolve that instead gets slightly different results

 References:
         G. Vivone, M. Dalla Mura, A. Garzelli, and F. Pacifici, "A Benchmarking Protocol for Pansharpening: Dataset, Pre-processing, and Quality Assessment", IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing, vol. 14, pp. 6102-6118, 2021.
         G. Vivone, M. Dalla Mura, A. Garzelli, R. Restaino, G. Scarpa, M. O. Ulfarsson, L. Alparone, and J. Chanussot, "A New Benchmark Based on Recent Advances in Multispectral Pansharpening: Revisiting Pansharpening With Classical and Emerging Pansharpening Methods", IEEE Geoscience and Remote Sensing Magazine, vol. 9, no. 1, pp. 53 - 81, March 2021.
"""
from scipy import ndimage
import numpy as np
from utils.sharpening_index.genMTF import genMTF, genMTF_torch


def MTF(I_MS, sensor, ratio):
    h = genMTF(ratio, sensor, I_MS.shape[2])

    I_MS_LP = np.zeros((I_MS.shape))
    for ii in range(I_MS.shape[2]):
        I_MS_LP[:, :, ii] = ndimage.correlate(
            I_MS[:, :, ii], h[:, :, ii], mode="nearest"
        )
        ### This can speed-up the processing, but with slightly different results with respect to the MATLAB toolbox
        # hb = h[:,:,ii]
        # I_MS_LP[:,:,ii] = signal.fftconvolve(I_MS[:,:,ii],hb[::-1],mode='same')

    return np.double(I_MS_LP)


# * ==========================================================
# * Pytorch version

import torch
import torch.nn.functional as F


def MTF_torch(fused: torch.Tensor, sensor: str, ratio: int):
    """torch version of MTF

    Args:
        fused (torch.Tensor): [B, C, H, W]
        sensor (str): sensor of the fused
        ratio (int): upsampled ratio from MS
    """
    nbands = fused.shape[1]
    h = genMTF_torch(ratio, sensor, nbands).permute(
        2, 0, 1
    )  # [k_h, k_w, c] -> [c, k_h, k_w]
    # Expand h to match batch dimension
    B = fused.shape[0]
    h = h.unsqueeze(0).expand(B, -1, -1, -1)  # [c, k_h, k_w] -> [B, c, k_h, k_w]
    # vmap on channels
    correlate_bs = torch.vmap(correlate_torch, in_dims=(0, 0), out_dims=0)
    correlate_bs_c = torch.vmap(correlate_bs, in_dims=(0, 0), out_dims=0)
    # vmap on batch
    I_MS_LP = correlate_bs_c(fused, h)

    return I_MS_LP


def correlate_torch(
    input: torch.Tensor, kernel: torch.Tensor, mode="nearest", constant_value=0
) -> torch.Tensor:
    """
    模拟 scipy.ndimage.filters.correlate 的功能，并支持多种边界填充模式。

    参数:
    - input: 输入张量 (形状: [batch_size, channels, height, width])
    - kernel: 卷积核 (形状: [height, width])
    - mode: 边界填充模式 ('nearest', 'reflect', 'constant', 'zeros')
    - constant_value: 当 mode='constant' 时使用的填充值

    返回:
    - 输出张量 (形状与输入相同)
    """
    # 确保输入和卷积核是浮点类型
    input = input.float()
    kernel = kernel.float()

    # 手动翻转卷积核以抵消 conv2d 的翻转
    kernel_flipped = torch.flip(kernel, dims=[0, 1])

    # 调整卷积核的形状为 [out_channels, in_channels, kernel_height, kernel_width]
    kernel_flipped = kernel_flipped.unsqueeze(0).unsqueeze(0)

    # 如果输入是单通道的，调整为 [batch_size=1, channels=1, height, width]
    if input.dim() == 2:  # 输入是 [height, width]
        input = input.unsqueeze(0).unsqueeze(0)
    elif input.dim() == 3:  # 输入是 [channels, height, width]
        input = input.unsqueeze(0)

    # 根据模式选择填充方式
    padding = kernel.shape[0] // 2  # 假设卷积核大小为奇数
    if mode == "nearest":
        input_padded = F.pad(
            input, (padding, padding, padding, padding), mode="replicate"
        )
    elif mode == "reflect":
        input_padded = F.pad(
            input, (padding, padding, padding, padding), mode="reflect"
        )
    elif mode == "constant":
        input_padded = F.pad(
            input,
            (padding, padding, padding, padding),
            mode="constant",
            value=constant_value,
        )
    elif mode == "zeros":
        input_padded = F.pad(
            input, (padding, padding, padding, padding), mode="constant", value=0
        )
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    # 计算输出大小相同的卷积
    output = F.conv2d(input_padded, kernel_flipped)

    # 如果输入是单通道的，去掉多余的维度
    if output.size(0) == 1 and output.size(1) == 1:
        output = output.squeeze(0).squeeze(0)

    return output


if __name__ == "__main__":
    # 测试代码
    # input_tensor = torch.randn(1, 1, 256, 256)
    # kernel = torch.tensor([[0, 1, 0],
    #                         [1, 0, 1],
    #                         [0, 1, 0]], dtype=torch.float32)
    # output_tensor = correlate_torch(input_tensor, kernel)
    # print(output_tensor)

    # input_np = input_tensor.numpy()
    # kernel = kernel.numpy()
    # output_np = ndimage.correlate(input_np[0, 0], kernel, mode='nearest')
    # print(output_np)

    from scipy.io import loadmat

    fused = loadmat(
        "/Data4/YuZhong/Flow_matching/OT-flow-matching/runs/pansharpening/gf2/2025-03-05 20:04:41_fm_OT_True/results/reduced/output_mulExm_1.mat"
    )["sr"]
    fused = torch.from_numpy(fused).permute(2, 0, 1).unsqueeze(0)
    sensor = "GF2"
    ratio = 4
    output = MTF_torch(fused, sensor, ratio)

    fused = fused.numpy()[0].transpose(1, 2, 0)
    output_np = MTF(fused, sensor, ratio)
    print(output_np)

    # comp
    print((output - torch.as_tensor(output_np.transpose(-1, 0, 1))[None]).abs().max())
