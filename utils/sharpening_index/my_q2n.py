# GPL License
# Copyright (C) UESTC
# All Rights Reserved
# @Author  : Xiao Wu
# @reference:
#
# 由于dat及其方差等数值舍入存在误差，最终结果有0.001左右的误差
import numpy as np
import torch
import math


def q2n(gt, x, q_blocks_size, q_shift):
    """ """
    if isinstance(gt, torch.Tensor):
        gt = gt.cpu().numpy()
        x = x.cpu().numpy()

    N1, N2, N3 = gt.shape  # 255 255 8
    size2 = q_blocks_size  # 32

    stepx = math.ceil(N1 / q_shift)  # 8
    stepy = math.ceil(N2 / q_shift)  # 8

    if stepy <= 0:
        stepy = 1
        stepx = 1

    est1 = (stepx - 1) * q_shift + q_blocks_size - N1  # 1
    est2 = (stepy - 1) * q_shift + q_blocks_size - N2  # 1
    # if np.sum(np.array([est1 != 0, est2 != 0])) > 0:
    # refref = np.zeros(shape=[N1+1, N2+1])
    # fusfus = refref.copy()
    if sum([(est1 != 0), (est2 != 0)]) > 0:
        for i in range(N3):
            a1 = gt[..., 0]

            ia1 = np.zeros(shape=[N1 + est1, N2 + est2])
            ia1[:N1, :N2] = a1
            ia1[:, N2 : N2 + est2] = ia1[:, N2 - 1 : -1 : N2 - est2 + 1]
            ia1[N1 : N1 + est1, ...] = ia1[N1 - 1 : -1 : N1 - est1 + 1, ...]
            if i == 0:
                refref = ia1[..., np.newaxis]  # np.concatenate(refref, ia1, axis=3)
            else:
                refref = np.concatenate([refref, ia1[..., np.newaxis]], axis=-1)
            if i < N3:
                gt = gt[..., 1:]

        gt = refref

        for i in range(N3):
            a2 = x[..., 0]
            ia2 = np.zeros(shape=[N1 + est1, N2 + est2])
            ia2[:N1, :N2] = a2
            ia2[:, N2 : N2 + est2] = ia2[:, N2 - 1 : -1 : N2 - est2 + 1]
            ia2[N1 : N1 + est1, ...] = ia2[N1 - 1 : -1 : N1 - est1 + 1, ...]
            if i == 0:
                fusfus = ia2[..., np.newaxis]  # np.concatenate(refref, ia1, axis=3)
            else:
                fusfus = np.concatenate([fusfus, ia2[..., np.newaxis]], axis=-1)

            if i < N3:
                x = x[..., 1:]
        x = fusfus

    # Equivalent to uint16(gt) in matlab (Xiao Wu in UESTC)
    x = np.round(x.clip(0)).astype(np.uint16)
    gt = np.round(gt.clip(0)).astype(np.uint16)

    N1, N2, N3 = gt.shape

    if math.ceil(math.log2(N3)) - math.log2(N3) != 0:
        Ndif = pow(2, math.ceil(math.log2(N3))) - N3
        dif = np.zeros(shape=[N1, N2, Ndif], dtype=np.uint16)
        gt = np.concatenate([gt, dif], axis=-1)
        x = np.concatenate([x, dif], axis=-1)

    _, _, N3 = gt.shape

    valori = np.zeros(shape=[stepx, stepy, N3])

    for j in range(stepx):
        for i in range(stepy):
            o = onions_quality(
                gt[
                    j * q_shift : j * q_shift + q_blocks_size,
                    i * q_shift : i * q_shift + size2,
                    :,
                ],
                x[
                    j * q_shift : j * q_shift + q_blocks_size,
                    i * q_shift : i * q_shift + size2,
                    :,
                ],
                q_blocks_size,
            )
            # 0.971379489438014	0.00553590637316723	0.00305237797490489	-0.0188289323262161	-0.00420556598390016	-0.0173947468044076	-0.0202144450367593	0.0102693855205061
            valori[j, i, :] = o
    q2n_idx_map = np.sqrt(np.sum(valori**2, axis=-1))
    q2n_index = np.mean(q2n_idx_map)

    return q2n_index, q2n_idx_map


def norm_blocco(x, eps=1e-8):
    a = x.mean()
    c = x.std()
    if c == 0:
        c = eps
    return (x - a) / c + 1, a, c


def onions_quality(dat1, dat2, size1):
    dat1 = np.float64(dat1)
    dat2 = np.float64(dat2)

    dat2 = np.concatenate([dat2[..., 0, np.newaxis], -dat2[..., 1:]], axis=-1)
    _, _, N3 = dat1.shape
    size2 = size1

    # Block norm
    """
            319.6474609375 37.05174450544686
            357.970703125 61.54042371537683
            518.708984375 111.53732768071865
            608.23828125 154.26606056030568
            593.412109375 163.97722215177643
            554.8486328125 113.96758695803403
            690.16015625 151.29524031046248
            442.2314453125 94.12877724960003
            mat
              319.6475   37.0698

              357.9707   61.5705

              518.7090  111.5918

              608.2383  154.3414

              593.4121  164.0573

              554.8486  114.0233

              690.1602  151.3692

              442.2314   94.1748
            """
    for i in range(N3):
        a1, s, t = norm_blocco(np.squeeze(dat1[..., i]))
        # print(s,t)
        dat1[..., i] = a1
        if s == 0:
            if i == 0:
                dat2[..., i] = dat2[..., i] - s + 1
            else:
                dat2[..., i] = -(-dat2[..., i] - s + 1)
        else:
            if i == 0:
                dat2[..., i] = ((dat2[..., i] - s) / t) + 1
            else:
                dat2[..., i] = -(((-dat2[..., i] - s) / t) + 1)
    m1 = np.zeros(shape=[N3])
    m2 = m1.copy()

    mod_q1m = 0
    mod_q2m = 0
    mod_q1 = np.zeros(shape=[size1, size2])
    mod_q2 = np.zeros(shape=[size1, size2])

    for i in range(N3):
        m1[..., i] = np.mean(np.squeeze(dat1[..., i]))
        m2[..., i] = np.mean(np.squeeze(dat2[..., i]))
        mod_q1m += m1[..., i] ** 2
        mod_q2m += m2[..., i] ** 2
        mod_q1 += np.squeeze(dat1[..., i]) ** 2
        mod_q2 += np.squeeze(dat2[..., i]) ** 2

    mod_q1m = np.sqrt(mod_q1m)
    mod_q2m = np.sqrt(mod_q2m)
    mod_q1 = np.sqrt(mod_q1)
    mod_q2 = np.sqrt(mod_q2)

    termine2 = mod_q1m * mod_q2m  # 7.97
    termine4 = mod_q1m**2 + mod_q2m**2  #
    int1 = (size1 * size2) / (size1 * size2 - 1) * np.mean(mod_q1**2)
    int2 = (size1 * size2) / (size1 * size2 - 1) * np.mean(mod_q2**2)
    termine3 = (
        int1 + int2 - (size1 * size2) / (size1 * size2 - 1) * (mod_q1m**2 + mod_q2m**2)
    )  # 17.8988  ** 2
    mean_bias = 2 * termine2 / termine4  # 1
    if termine3 == 0:
        q = np.zeros(shape=[1, N3])
        q[:, :, N3 - 1] = mean_bias
    else:
        cbm = 2 / termine3
        # 32 32 8
        qu = onion_mult2D(dat1, dat2)
        qm = onion_mult(m1.reshape(-1), m2.reshape(-1))
        qv = np.zeros(shape=[N3])
        for i in range(N3):
            qv[i] = (
                (size1 * size2)
                / ((size1 * size2) - 1)
                * np.mean(np.squeeze(qu[:, :, i]))
            )
        q = qv - (size1 * size2) / ((size1 * size2) - 1) * qm
        q = q * mean_bias * cbm

    return q


def onion_mult2D(onion1, onion2):
    _, _, N3 = onion1.shape

    if N3 > 1:
        L = N3 // 2
        a = onion1[..., :L]
        b = onion1[..., L:]
        b = np.concatenate([b[..., 0, np.newaxis], -b[..., 1:]], axis=-1)
        c = onion2[..., :L]
        d = onion2[..., L:]
        d = np.concatenate([d[..., 0, np.newaxis], -d[..., 1:]], axis=-1)

        if N3 == 2:
            ris = np.concatenate([a * c - d * b, a * d + c * b], axis=-1)
        else:
            ris1 = onion_mult2D(a, c)
            ris2 = onion_mult2D(
                d, np.concatenate([b[..., 0, np.newaxis], -b[..., 1:]], axis=-1)
            )
            ris3 = onion_mult2D(
                np.concatenate([a[..., 0, np.newaxis], -a[..., 1:]], axis=-1), d
            )
            ris4 = onion_mult2D(c, b)

            aux1 = ris1 - ris2
            aux2 = ris3 + ris4

            ris = np.concatenate([aux1, aux2], axis=-1)
    else:
        ris = onion1 * onion2
    return ris


def onion_mult(onion1, onion2):
    # _, N = onion1.shape
    N = len(onion1)
    if N > 1:
        L = N // 2
        a = onion1[:L]
        b = onion1[L:]
        # b[1:] = -b[1:]
        b = np.append(np.array(b[0]), -b[1:])
        c = onion2[:L]
        d = onion2[L:]
        # d[1:] = -d[1:]
        d = np.append(np.array(d[0]), -d[1:])

        if N == 2:
            ris = np.append(a * c - d * b, a * d + c * b)
        else:
            ris1 = onion_mult(a, c)
            # b[1:] = -b[1:]
            ris2 = onion_mult(d, np.append(np.array(b[0]), -b[1:]))
            # a[1:] = -a[1:]
            ris3 = onion_mult(np.append(np.array(a[0]), -a[1:]), d)
            ris4 = onion_mult(c, b)

            aux1 = ris1 - ris2
            aux2 = ris3 + ris4
            ris = np.append(aux1, aux2)
    else:
        ris = np.array(onion1).reshape(-1) * np.array(onion2).reshape(-1)
    return ris


# * ==========================================================
# * Torch version


def qnr_torch(
    gt,
    x,
    q_blocks_size,
    q_shift,
):
    N3, N1, N2 = gt.size()  # [c, h, w]
    size2 = q_blocks_size

    stepx = math.ceil(N1 / q_shift)  # 8
    stepy = math.ceil(N2 / q_shift)  # 8

    if stepy <= 0:
        stepy = 1
        stepx = 1

    est1 = (stepx - 1) * q_shift + q_blocks_size - N1  # 1
    est2 = (stepy - 1) * q_shift + q_blocks_size - N2  # 1

    if (est1 != 0) or (est2 != 0):

        def channel_gt_x(gt_):
            a1 = gt_[0]
            ia1 = torch.zeros(N1 + est1, N2 + est2)
            ia1[:N1, :N2] = a1
            ia1[..., N2 : N2 + est2] = ia1[..., N2 - 1 : -1 : N2 - est2 + 1]
            ia1[..., N1 : N1 + est1] = ia1[..., N1 - 1 : -1 : N1 - est1 + 1]
            return ia1

        # for loops
        ia1_s = []
        ia2_s = []
        gt_ = gt.clone()
        x_ = x.clone()
        for idx in range(N3):
            ia1 = channel_gt_x(gt_)
            ia1_s.append(ia1)

            ia2 = channel_gt_x(x_)
            ia2_s.append(ia2)

            if idx < N3:
                gt_ = gt_[1:]
                x_ = x_[1:]

        gt = torch.stack(ia1_s, dim=0)
        x = torch.stack(ia2_s, dim=0)

    x = torch.round(x.clamp(0)).to(torch.uint16)
    gt = torch.round(gt.clamp(0)).to(torch.uint16)

    if math.ceil(math.log2(N3)) - math.log2(N3) != 0:
        Ndif = pow(2, math.ceil(math.log2(N3))) - N3
        dif = torch.zeros(Ndif, N1, N2, dtype=torch.uint16)
        gt = torch.cat([gt, dif], dim=0)
        x = torch.cat([x, dif], dim=0)

    N3 = gt.shape[0]

    valori = torch.zeros(N3, stepx, stepy)
    for j in range(stepx):
        for i in range(stepy):
            # FIXME: to reimplement the function
            valori[:, j, i] = onions_quality(
                gt[
                    :,
                    j * q_shift : j * q_shift + q_blocks_size,
                    i * q_shift : i * q_shift + size2,
                ],
                x[
                    :,
                    j * q_shift : j * q_shift + q_blocks_size,
                    i * q_shift : i * q_shift + size2,
                ],
            )
    q2n_idx_map = torch.sqrt(torch.sum(valori**2, dim=-1))
    q2n_index = torch.mean(q2n_idx_map)

    return q2n_index, q2n_idx_map


def norm_blocco_torch(x, eps=1e-8):
    a = x.mean()
    c = x.std()
    if c == 0:
        c = eps
    return (x - a) / c + 1, a, c


def onion_mult_2D_torch(onion1: torch.Tensor, onion2: torch.Tensor) -> torch.Tensor:
    *_, N3 = onion1.size()
    if N3 > 1:
        L = N3 // 2
        a = onion1[..., :L]
        b = onion1[..., L:]
        b = torch.cat([b[..., 0:1], -b[..., 1:]], dim=-1)
        c = onion2[..., :L]
        d = onion2[..., L:]
        d = torch.cat([d[..., 0:1], -d[..., 1:]], dim=-1)

        if N3 == 2:
            ris = torch.cat([a * c - d * b, a * d + c * b], dim=-1)
        else:
            ris1 = onion_mult_2D_torch(a, c)
            ris2 = onion_mult_2D_torch(
                d,
                torch.cat([b[..., 0:1], -b[..., 1:]], dim=-1),
            )
            ris3 = onion_mult_2D_torch(
                torch.cat([a[..., 0:1], -a[..., 1:]], dim=-1),
                d,
            )
            ris4 = onion_mult_2D_torch(c, b)

            aux1 = ris1 - ris2
            aux2 = ris3 + ris4

            ris = torch.cat([aux1, aux2], dim=-1)
    else:
        ris = onion1 * onion2

    return ris


def onion_mult_torch(onion1: torch.Tensor, onion2: torch.Tensor) -> torch.Tensor:
    """
    计算两个洋葱数的乘积，使用 PyTorch 实现。

    参数:
    - onion1: 第一个洋葱数 (形状: [N])
    - onion2: 第二个洋葱数 (形状: [N])

    返回:
    - 结果洋葱数 (形状: [N])
    """
    N = onion1.size(0)  # 获取洋葱数的维度

    if N > 1:
        L = N // 2

        # 分割洋葱数
        a = onion1[:L]
        b = onion1[L:]
        c = onion2[:L]
        d = onion2[L:]

        # 对 b 和 d 的后半部分取负值
        b = torch.cat([b[:1], -b[1:]])
        d = torch.cat([d[:1], -d[1:]])

        if N == 2:
            # 基础情况：复数乘法
            ris = torch.cat([a * c - d * b, a * d + c * b])
        else:
            # 递归计算子问题
            ris1 = onion_mult_torch(a, c)
            ris2 = onion_mult_torch(d, torch.cat([b[:1], -b[1:]]))
            ris3 = onion_mult_torch(torch.cat([a[:1], -a[1:]]), d)
            ris4 = onion_mult_torch(c, b)

            # 组合结果
            aux1 = ris1 - ris2
            aux2 = ris3 + ris4
            ris = torch.cat([aux1, aux2])
    else:
        # 基础情况：标量乘法
        ris = onion1.view(-1) * onion2.view(-1)

    return ris


# * ==========================================================
# * tests


def test_onion_mult():
    a = torch.randn(16)
    b = torch.randn(16)
    res = onion_mult_torch(a, b)

    a = a.numpy()
    b = b.numpy()
    res_np = onion_mult(a, b)

    print(res)
    print(res_np)


def test_onion_mult_2D():
    a = torch.randn(16, 16, 8)
    b = torch.randn(16, 16, 8)
    res = onion_mult_2D_torch(a, b)

    a = a.numpy()
    b = b.numpy()
    res_np = onion_mult2D(a, b)

    print(res)
    print(res_np)

    print(np.max(np.abs(res.numpy() - res_np)))


if __name__ == "__main__":
    # test_onion_mult()
    # test_onion_mult_2D()
    x = torch.randn(8, 256, 256)
    gt = torch.randn(8, 256, 256)
    qnr_result = qnr_torch(gt, x, 32, 8)
    print("QNR Result:", qnr_result)
