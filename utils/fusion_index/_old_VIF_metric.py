import torch
from kornia.color.ycbcr import rgb_to_y
from torch import Tensor
from torch.nn import functional as F
from torchmetrics.functional.image import visual_information_fidelity
from tqdm import tqdm
from typing_extensions import TypeAlias

########## ================================== legacy code ================================== ##########
# source code from xiao-woo
#! old previous callable functions
#########
# metric helpers
#########


def cal_PSNR(A, B, F):
    [m, n] = F.shape
    MSE_AF = torch.sum((F - A) ** 2) / (m * n)
    MSE_BF = torch.sum((F - B) ** 2) / (m * n)
    MSE = 0.5 * MSE_AF + 0.5 * MSE_BF
    PSNR = 20 * torch.log10(255 / torch.sqrt(MSE))

    return PSNR


def cal_SD(F):
    [m, n] = F.shape
    u = torch.mean(F)

    # 原版 is wrong
    # tmp = (F - u).numpy()
    # tmp2 = np.round(tmp.clip(0)).astype(np.uint16)
    # SD = np.sqrt(np.sum((tmp2 ** 2).clip(0, 255)) / (m * n))

    SD = torch.sqrt(torch.sum((F - u) ** 2) / (m * n))

    return SD


def cal_VIF(A: Tensor, B: Tensor, F: Tensor):
    def check_dims(x):
        ndim = x.ndim
        if ndim == 3:  # [c, h, w]
            x = x[None]
        elif ndim == 2:
            x = x[None, None]
        elif ndim == 4:
            pass
        else:
            raise ValueError(f"x.ndim should be 2 or 3, but got {ndim}")

        return x

    A, B, F = map(check_dims, (A, B, F))

    fusion_chans = F.size(1)
    if A.size(1) != fusion_chans:
        assert A.size(1) == 1
        A = A.expand_as(F)
    if B.size(1) != fusion_chans:
        assert B.size(1) == 1
        B = B.expand_as(F)

    # x: [b, c, h, w]
    assert A.size(1) == B.size(1) == F.size(1), (
        f"A.size(1) should be equal to B.size(1) and F.size(1), "
        f"but got A.size(1)={A.size(1)}, B.size(1)={B.size(1)}, F.size(1)={F.size(1)}"
    )

    vif_fn = visual_information_fidelity
    return vif_fn(F, A) + vif_fn(F, B)


def cal_EN(I):
    p = torch.histc(I, 256)
    p = p[p != 0]
    p = p / torch.numel(I)
    E = -torch.sum(p * torch.log2(p))
    return E


def cal_SF(MF):
    # RF = MF[]#diff(MF, 1, 1);
    [m, n] = MF.shape
    RF = MF[: m - 1, :] - MF[1:]
    RF1 = torch.sqrt(torch.mean(RF**2))
    CF = MF[:, : n - 1] - MF[:, 1:]  # diff(MF, 1, 2)
    CF1 = torch.sqrt(torch.mean(CF**2))
    SF = torch.sqrt(RF1**2 + CF1**2)

    return SF


def analysis_Reference_fast(
    image_vis: "Tensor",
    image_ir: "Tensor",
    image_f: "Tensor",
):
    # shapes are [c, h, w], channel is 1 or 3
    # image_f: 0-255
    # image_ir: 0-255
    # image_vis: 0-255

    if image_f.ndim == 2:
        PSNR = cal_PSNR(image_ir, image_vis, image_f)
        SD = cal_SD(image_f)
        EN = cal_EN(image_f)
        SF = cal_SF(image_f / 255.0)
        AG = cal_AG(image_f)
        SSIM = cal_SSIM(image_ir, image_vis, image_f)
    elif image_f.ndim == 3:  # [c, h, w]
        # compute per channel
        fusion_chans = image_f.size(0)
        assert image_ir.ndim == 2, "force to expand ir to fusion channel"
        image_ir = image_ir[None].expand_as(image_f)  # [c, h, w]
        PSNRs, SDs, ENs, SFs, AGs, SSIMs = [], [], [], [], [], []
        for i in range(fusion_chans):
            PSNR = cal_PSNR(image_ir[i], image_vis[i], image_f[i])
            SD = cal_SD(image_f[i])
            EN = cal_EN(image_f[i])
            SF = cal_SF(image_f[i] / 255.0)
            AG = cal_AG(image_f[i])
            SSIM = cal_SSIM(image_ir[i], image_vis[i], image_f[i])

            PSNRs.append(PSNR)
            SDs.append(SD)
            ENs.append(EN)
            SFs.append(SF)
            AGs.append(AG)
            SSIMs.append(SSIM)
        PSNR, SD, EN, SF, AG, SSIM = [
            sum(x) / len(x) for x in [PSNRs, SDs, ENs, SFs, AGs, SSIMs]
        ]

    # taken batched tensors (batching inside the function)
    VIF = cal_VIF(image_ir, image_vis, image_f)

    return dict(
        PSNR=PSNR.item(),
        EN=EN.item(),
        SD=SD.item(),
        SF=SF.item(),
        AG=AG.item(),
        SSIM=SSIM.item(),
        VIF=VIF.item(),
    )


def cal_AG(img):
    if len(img.shape) == 2:
        [r, c] = img.shape
        [dzdx, dzdy] = torch.gradient(img)
        s = torch.sqrt((dzdx**2 + dzdy**2) / 2)
        g = torch.sum(s) / ((r - 1) * (c - 1))

    else:
        [r, c, b] = img.shape
        g = torch.zeros(b)
        for k in range(b):
            band = img[:, :, k]
            [dzdx, dzdy] = torch.gradient(band)
            s = torch.sqrt((dzdx**2 + dzdy**2) / 2)
            g[k] = torch.sum(s) / ((r - 1) * (c - 1))
    return torch.mean(g)


def _ssim(img1, img2):
    device = img1.device
    img1 = img1.float()
    img2 = img2.float()

    channel = img1.shape[1]
    max_val = 1
    _, c, w, h = img1.size()
    window_size = min(w, h, 11)
    sigma = 1.5 * window_size / 11

    # 不加这个,对应matlab的quality_assess的ssim指标
    # pad_size = [window_size//2]*4
    # img1 = F.pad(img1, mode='replicate', pad=pad_size)
    # img2 = F.pad(img2, mode='replicate', pad=pad_size)

    window = create_window(window_size, sigma, channel).to(device)
    mu1 = F.conv2d(img1, window, groups=channel)  # , padding=window_size // 2
    mu2 = F.conv2d(img2, window, groups=channel)  # , padding=window_size // 2

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = (
        F.conv2d(img1 * img1, window, groups=channel) - mu1_sq
    )  # , padding=window_size // 2
    sigma2_sq = (
        F.conv2d(img2 * img2, window, groups=channel) - mu2_sq
    )  # , padding=window_size // 2
    sigma12 = (
        F.conv2d(img1 * img2, window, groups=channel) - mu1_mu2
    )  # , padding=window_size // 2
    C1 = (0.01 * max_val) ** 2
    C2 = (0.03 * max_val) ** 2
    V1 = 2.0 * sigma12 + C2
    V2 = sigma1_sq + sigma2_sq + C2
    ssim_map = ((2 * mu1_mu2 + C1) * V1) / ((mu1_sq + mu2_sq + C1) * V2)
    t = ssim_map.shape
    return ssim_map.mean(2).mean(2)


import math

from torch.autograd import Variable


def gaussian(window_size, sigma):
    gauss = torch.Tensor(
        [
            math.exp(-((x - window_size // 2) ** 2) / float(2 * sigma**2))
            for x in range(window_size)
        ]
    )
    return gauss / gauss.sum()


def create_window(window_size, sigma, channel):
    _1D_window = gaussian(window_size, sigma).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(
        _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    )
    return window


def cal_SSIM(im1, im2, image_f):
    # h, w -> 2, h, w -> b, 2, h, w
    img_Seq = torch.stack([im1, im2])
    image_f = image_f.unsqueeze(0).repeat([img_Seq.shape[0], 1, 1])
    return torch.mean(_ssim(img_Seq.unsqueeze(0) / 255.0, image_f / 255.0))
