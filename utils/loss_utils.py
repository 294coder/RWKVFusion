import inspect
import math
import random
import sys
from functools import partial
from math import exp
from typing import TYPE_CHECKING, Sequence, Union

import kornia
import lpips
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from kornia.filters import laplacian, spatial_gradient
from torch import Tensor
from torch.autograd import Variable

from utils._ydtr_loss import sf_loss_ir, sf_loss_vi, ssim_loss_ir, ssim_loss_vi
from utils.log_utils import easy_logger
from utils.misc import default, exists, is_main_process, is_none
from utils.torch_dct import dct_2d, idct_2d
from utils.vgg import vgg16

if TYPE_CHECKING:
    from model.base_model import BaseModel

logger = easy_logger(func_name="loss_utils")

######################### helper functions #########################


def accum_loss_dict(ep_loss_dict: dict, loss_dict: dict):
    for k, v in loss_dict.items():
        if k in ep_loss_dict:
            ep_loss_dict[k] += v
        else:
            ep_loss_dict[k] = v
    return ep_loss_dict


def ave_ep_loss(ep_loss_dict: dict, ep_iters: int):
    assert ep_iters > 0, "ep_iters must be greater than 0"
    for k, v in ep_loss_dict.items():
        ep_loss_dict[k] = v / ep_iters
    return ep_loss_dict


# @is_main_process
def ave_multi_rank_dict(dict_metrics: "list[dict] | dict", one_rank_zero: bool = True):
    if one_rank_zero and is_main_process():
        # type is dict is only one process
        assert isinstance(
            dict_metrics, (list, dict)
        ), "rank_loss_dict must be a list or a dict"

        if isinstance(dict_metrics, dict):
            return dict_metrics

        n = len(dict_metrics)
        if n == 1:
            return dict_metrics[0]

        ave_dict = {}
        keys = dict_metrics[0].keys()

        for k in keys:
            vs = 0
            for d in dict_metrics:
                v = d[k]
                vs = vs + v
            ave_dict[k] = vs / n
        return ave_dict
    else:
        return dict_metrics


####################################################################


###########################################################################
########### Multi-task Gradient Normalization Scale #######################
# Ref: taken from Simo Ryu's trick for multi-task learning, thanks.


class GradientNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x

    @staticmethod
    def backward(ctx, grad_output):
        grad_output_norm = torch.norm(grad_output, p=2)
        eps = 1e-8
        if grad_output_norm == 0:
            grad_output_norm = grad_output
        else:
            grad_output_norm = grad_output / (grad_output_norm + eps)

        return grad_output_norm


def grad_norm(x):
    return GradientNormFunction.apply(x)


###########################################################################


class PerceptualLoss(nn.Module):
    def __init__(self, percep_net="vgg", norm=True):
        super(PerceptualLoss, self).__init__()
        self.norm = norm
        self.lpips_loss = lpips.LPIPS(net=percep_net).cuda()

    def forward(self, x, y):
        # assert x.shape == y.shape
        loss = self.lpips_loss(x, y, normalize=self.norm)
        return torch.squeeze(loss).mean()


def gaussian(window_size, sigma):
    gauss = torch.Tensor(
        [
            exp(-((x - window_size // 2) ** 2) / float(2 * sigma**2))
            for x in range(window_size)
        ]
    )
    return gauss / gauss.sum()


class MaxGradientLoss(torch.nn.Module):
    def __init__(self, mean_batch=True) -> None:
        super().__init__()
        self.register_buffer(
            "x_sobel_kernel",
            torch.FloatTensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).expand(1, 1, 3, 3),
        )
        self.register_buffer(
            "y_sobel_kernel",
            torch.FloatTensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]).expand(1, 1, 3, 3),
        )
        self.mean_batch = mean_batch

    def forward(self, fuse, ir, vis):
        c = fuse.size(1)

        fuse_grad_x = F.conv2d(fuse, self.x_sobel_kernel, padding=1, groups=c)
        fuse_grad_y = F.conv2d(fuse, self.y_sobel_kernel, padding=1, groups=c)

        ir_grad_x = F.conv2d(ir, self.x_sobel_kernel, padding=1, groups=c)
        ir_grad_y = F.conv2d(ir, self.y_sobel_kernel, padding=1, groups=c)

        vis_grad_x = F.conv2d(vis, self.x_sobel_kernel, padding=1, groups=c)
        vis_grad_y = F.conv2d(vis, self.y_sobel_kernel, padding=1, groups=c)

        max_grad_x = torch.maximum(ir_grad_x, vis_grad_x)
        max_grad_y = torch.maximum(ir_grad_y, vis_grad_y)

        if self.mean_batch:
            max_gradient_loss = (
                F.l1_loss(fuse_grad_x, max_grad_x) + F.l1_loss(fuse_grad_y, max_grad_y)
            ) / 2
        else:
            x_loss_b = F.l1_loss(fuse_grad_x, max_grad_x, reduction="none").mean(
                dim=(1, 2, 3)
            )
            y_loss_b = F.l1_loss(fuse_grad_y, max_grad_y, reduction="none").mean(
                dim=(1, 2, 3)
            )

            max_gradient_loss = (x_loss_b + y_loss_b) / 2

        return max_gradient_loss


def create_window(window_size, channel, sigma=1.5):
    _1D_window = gaussian(window_size, sigma).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(
        _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    )
    return window


def mci_loss(pred, gt):
    return F.l1_loss(pred, gt.max(1, keepdim=True)[0])


def sf(f1, kernel_radius=5):
    """copy from https://github.com/tthinking/YDTR/blob/main/losses/__init__.py

    Args:
        f1 (torch.Tensor): image shape [b, c, h, w]
        kernel_radius (int, optional): kernel redius using calculate sf. Defaults to 5.

    Returns:
        loss: loss item. type torch.Tensor
    """

    device = f1.device
    b, c, h, w = f1.shape
    r_shift_kernel = (
        torch.FloatTensor([[0, 0, 0], [1, 0, 0], [0, 0, 0]])
        .to(device)
        .reshape((1, 1, 3, 3))
        .repeat(c, 1, 1, 1)
    )
    b_shift_kernel = (
        torch.FloatTensor([[0, 1, 0], [0, 0, 0], [0, 0, 0]])
        .to(device)
        .reshape((1, 1, 3, 3))
        .repeat(c, 1, 1, 1)
    )
    f1_r_shift = F.conv2d(f1, r_shift_kernel, padding=1, groups=c)
    f1_b_shift = F.conv2d(f1, b_shift_kernel, padding=1, groups=c)
    f1_grad = torch.pow((f1_r_shift - f1), 2) + torch.pow((f1_b_shift - f1), 2)
    kernel_size = kernel_radius * 2 + 1
    add_kernel = torch.ones((c, 1, kernel_size, kernel_size)).float().to(device)
    kernel_padding = kernel_size // 2
    f1_sf = torch.sum(
        F.conv2d(f1_grad, add_kernel, padding=kernel_padding, groups=c), dim=1
    )
    return 1 - f1_sf


class HybridL1L2(torch.nn.Module):
    def __init__(self):
        super(HybridL1L2, self).__init__()
        self.l1 = torch.nn.L1Loss()
        self.l2 = torch.nn.MSELoss()
        self.loss = LossWarpper(l1=self.l1, l2=self.l2)

    def forward(self, pred, gt):
        loss, loss_dict = self.loss(pred, gt)
        return loss, loss_dict


class HybridSSIMSF(torch.nn.Module):
    def __init__(self, channel, weighted_r=(1.0, 5e-2, 6e-4, 25e-5)) -> None:
        super().__init__()
        self.weighted_r = weighted_r

    def forward(self, fuse, gt):
        # fuse: [b, 1, h, w]
        vi = gt[:, 0:1]  # [b, 1, h, w]
        ir = gt[:, 1:]  # [b, 1, h, w]

        _ssim_f_ir = ssim_loss_ir(fuse, ir)
        _ssim_f_vi = ssim_loss_vi(fuse, vi)
        _sf_f_ir = sf_loss_ir(fuse, ir)
        _sf_f_vi = sf_loss_vi(fuse, vi)

        ssim_f_ir = self.weighted_r[0] * _ssim_f_ir
        ssim_f_vi = self.weighted_r[1] * _ssim_f_vi
        sf_f_ir = self.weighted_r[2] * _sf_f_ir
        sf_f_vi = self.weighted_r[3] * _sf_f_vi

        loss_dict = dict(
            ssim_f_ir=ssim_f_ir,
            ssim_f_vi=ssim_f_vi,
            sf_f_ir=sf_f_ir,
            sf_f_vi=sf_f_vi,
        )

        loss = ssim_f_ir + ssim_f_vi + sf_f_ir + sf_f_vi
        return loss, loss_dict


class HybridSSIMMCI(torch.nn.Module):
    def __init__(self, channel, weight_r=(1.0, 1.0, 1.0)) -> None:
        super().__init__()
        self.ssim = SSIMLoss(channel=channel)
        self.mci_loss = mci_loss
        self.weight_r = weight_r

    def forward(self, fuse, gt):
        # fuse: [b, 1, h, w]
        vi = gt[:, 0:1]  # [b, 1, h, w]
        ir = gt[:, 1:]  # [b, 1, h, w]

        _ssim_f_ir = self.weight_r[0] * self.ssim(fuse, ir)
        _ssim_f_vi = self.weight_r[1] * self.ssim(fuse, vi)
        _mci_loss = self.weight_r[2] * self.mci_loss(fuse, gt)

        loss = _ssim_f_ir + _ssim_f_vi + _mci_loss

        loss_dict = dict(
            ssim_f_ir=_ssim_f_ir,
            ssim_f_vi=_ssim_f_vi,
            mci_loss=_mci_loss,
        )

        return loss, loss_dict


################### SSIM Loss (different implementations) ##############

from kornia.losses import SSIMLoss as K_SSIMLoss


class SSIMLoss(torch.nn.Module):
    def __init__(
        self, window_size=11, win_sigma=1.5, data_range=1, size_average=True, channel=3
    ):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = channel
        self.window = create_window(window_size, self.channel, win_sigma)
        self.win_sigma = win_sigma

    def forward(self, img1, img2):
        # print(img1.size())
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel, self.win_sigma)

            if img1.is_cuda:
                window = window.to(img1.device)
            window = window.type_as(img1)

            self.window = window
            self.channel = channel

        return 1 - self._ssim(
            img1, img2, window, self.window_size, channel, self.size_average
        )

    @classmethod
    def ssim_map(cls, img1, img2, win_size=11, data_range=1, size_average=True):
        (_, channel, _, _) = img1.size()
        window = create_window(win_size, channel)

        if img1.is_cuda:
            window = window.to(img1.device)
        window = window.type_as(img1)

        return cls._ssim(img1, img2, window, win_size, channel, size_average)

    @classmethod
    def _ssim(cls, img1, img2, window, window_size, channel, size_average=True):
        mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = (
            F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel)
            - mu1_sq
        )
        sigma2_sq = (
            F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel)
            - mu2_sq
        )
        sigma12 = (
            F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel)
            - mu1_mu2
        )

        C1 = 0.01**2
        C2 = 0.03**2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
            (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
        )

        if size_average:
            return ssim_map.nanmean()
        else:
            return ssim_map.mean(1).mean(1).mean(1)


def get_ssim_loss(implem_by: str = "kornia", **ssim_kwargs):
    """
    get ssim loss by different implementations

    Args:
        implem_by: str, 'kornia' or 'torch'
        ssim_kwargs: dict, parameters for ssim loss

    Init SSIMLoss Kwargs:
        - kornia:
            window_size: int
            max_val: float = 1.0
            eps: float = 1e-12
            reduction: str = "mean"
            padding: str = "same"
        - torch:
            win_size: int = 11
            win_sigma: float = 1.5
            data_range: int = 1
            size_average: bool = True
            channel: int = 3

    Examples:
        >>> get_ssim_loss(
        ...     implem_by="kornia",
        ...     window_size=11,
        ...     max_val=1.0,
        ...     eps=1e-12,
        ...     reduction="mean",
        ...     padding="same",
        ... )
        >>> get_ssim_loss(
        ...     implem_by="torch",
        ...     win_size=11,
        ...     win_sigma=1.5,
        ...     data_range=1,
        ...     size_average=True,
        ...     channel=3,
        ... )
    """
    if implem_by == "kornia":
        ssim_kwargs.pop("channel", None)
        return K_SSIMLoss(**ssim_kwargs)
    elif implem_by == "torch":
        return SSIMLoss(**ssim_kwargs)
    else:
        raise ValueError(f"Invalid implementation choice: {implem_by}")


class HybridL1SSIM(torch.nn.Module):
    def __init__(
        self,
        channel: int = 31,
        weighted_r: Sequence[float] = (1.0, 0.1),
        grad_norm: bool = False,
        **ssim_kwargs,
    ):
        super(HybridL1SSIM, self).__init__()
        assert len(weighted_r) == 2
        self.grad_norm = grad_norm
        self._l1 = torch.nn.L1Loss(reduction="none")
        self._ssim = get_ssim_loss(**ssim_kwargs)
        if grad_norm:
            weighted_r = (1.0, 1.0)
        self.loss = LossWarpper(weighted_r, l1=self._l1, ssim=self._ssim)
        logger.info(
            f"weighted_r: {weighted_r} for L1 SSIM loss, grad_norm: {grad_norm}"
        )

    def forward(self, pred, gt):
        if self.grad_norm:
            pred = grad_norm(pred)
        loss, loss_dict = self.loss(pred, gt)
        return loss, loss_dict


class HybridCharbonnierSSIM(torch.nn.Module):
    def __init__(self, weighted_r, channel=31) -> None:
        super().__init__()
        self._ssim = SSIMLoss(channel=channel)
        self._charb = CharbonnierLoss(eps=1e-4)
        self.loss = LossWarpper(weighted_r, charbonnier=self._charb, ssim=self._ssim)

    def forward(self, pred, gt):
        loss, loss_dict = self.loss(pred, gt)
        return (loss,)


class HybridMCGMCI(torch.nn.Module):
    def __init__(self, weight_r=(1.0, 1.0)) -> None:
        super().__init__()
        self.mcg = MaxGradientLoss()
        self.mci = mci_loss
        self.weight_r = weight_r

    def forward(self, pred, gt):
        vis = gt[:, 0:1]
        ir = gt[:, 1:]

        mcg_loss = self.mcg(pred, ir, vis) * self.weight_r[0]
        mci_loss = self.mci(pred, gt) * self.weight_r[1]

        loss_dict = dict(mcg=mcg_loss, mci=mci_loss)

        return mcg_loss + mci_loss, loss_dict


def gradient(input):
    filter1 = nn.Conv2d(
        kernel_size=3, in_channels=1, out_channels=1, bias=False, padding=1, stride=1
    )
    filter2 = nn.Conv2d(
        kernel_size=3, in_channels=1, out_channels=1, bias=False, padding=1, stride=1
    )
    filter1.weight.data = (
        torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]])
        .reshape(1, 1, 3, 3)
        .to(input.device)
    )
    filter2.weight.data = (
        torch.tensor([[1.0, 2.0, 1.0], [0.0, 0.0, 0.0], [-1.0, -2.0, -1.0]])
        .reshape(1, 1, 3, 3)
        .to(input.device)
    )

    g1 = filter1(input)
    g2 = filter2(input)
    image_gradient = torch.abs(g1) + torch.abs(g2)
    return image_gradient


class LossWarpper(torch.nn.Module):
    def __init__(
        self,
        weighted_ratio=(1.0, 1.0),
        **losses: "dict[str, torch.nn.Module | callable]",
    ):
        super(LossWarpper, self).__init__()
        self.names = []
        assert len(weighted_ratio) == len(
            losses.keys()
        ), "`weighted_ratio` must be the same length as `losses`"
        self.weighted_ratio = weighted_ratio
        for k, v in losses.items():
            self.names.append(k)
            setattr(self, k, v)

    def forward(self, pred, gt) -> tuple[torch.Tensor, dict[torch.Tensor]]:
        loss = 0.0
        d_loss = {}
        for i, n in enumerate(self.names):
            l = getattr(self, n)(pred, gt) * self.weighted_ratio[i]
            if l.numel() != 1:
                l = l.nanmean()
            loss += l
            d_loss[n] = l
        return loss, d_loss


class TorchLossWrapper(torch.nn.Module):
    def __init__(
        self,
        weight_ratio: Union[tuple[float], list[float]],
        **loss: "dict[str, torch.nn.Module | callable]",
    ) -> None:
        super().__init__()
        self.key = list(loss.keys())
        self.loss = list(loss.values())
        self.weight_ratio = weight_ratio

        assert len(weight_ratio) == len(loss.keys())

    def forward(self, pred, gt):
        loss_total = 0.0
        loss_d = {}
        for i, l in enumerate(self.loss):
            loss_i = l(pred, gt) * self.weight_ratio[i]
            loss_total = loss_total + loss_i

            k = self.key[i]
            loss_d[k] = loss_i

        return loss_total, loss_d


def elementwise_charbonnier_loss(
    input: Tensor, target: Tensor, eps: float = 1e-3
) -> Tensor:
    """Apply element-wise weight and reduce loss between a pair of input and
    target.
    """
    return torch.sqrt((input - target) ** 2 + (eps * eps))


class HybridL1L2(nn.Module):
    def __init__(self, cof=10.0):
        super(HybridL1L2, self).__init__()
        self.l1 = nn.L1Loss()
        self.l2 = nn.MSELoss()
        self.cof = cof

    def forward(self, pred, gt):
        return self.l1(pred, gt) / self.cof + self.l2(pred, gt)


class RMILoss(nn.Module):
    def __init__(
        self,
        with_logits=False,
        radius=3,
        bce_weight=0.5,
        downsampling_method="max",
        stride=3,
        use_log_trace=True,
        use_double_precision=True,
        epsilon=0.0005,
    ):
        super().__init__()

        self.use_double_precision = use_double_precision
        self.with_logits = with_logits
        self.bce_weight = bce_weight
        self.stride = stride
        self.downsampling_method = downsampling_method
        self.radius = radius
        self.use_log_trace = use_log_trace
        self.epsilon = epsilon

    def forward(self, input, target):
        if self.bce_weight != 0:
            if self.with_logits:
                bce = F.binary_cross_entropy_with_logits(input, target=target)
            else:
                bce = F.binary_cross_entropy(input, target=target)
            bce = bce.mean() * self.bce_weight
        else:
            bce = 0.0

        if self.with_logits:
            input = torch.sigmoid(input)

        rmi = self.rmi_loss(input=input, target=target)
        rmi = rmi.mean() * (1.0 - self.bce_weight)
        return rmi + bce

        return bce

    def rmi_loss(self, input, target):
        assert input.shape == target.shape
        vector_size = self.radius * self.radius

        y = self.extract_region_vector(target)
        p = self.extract_region_vector(input)

        if self.use_double_precision:
            y = y.double()
            p = p.double()

        eps = torch.eye(vector_size, dtype=y.dtype, device=y.device) * self.epsilon
        eps = eps.unsqueeze(dim=0).unsqueeze(dim=0)

        y = y - y.mean(dim=3, keepdim=True)
        p = p - p.mean(dim=3, keepdim=True)

        y_cov = y @ self.transpose(y)
        p_cov = p @ self.transpose(p)
        y_p_cov = y @ self.transpose(p)

        m = y_cov - y_p_cov @ self.transpose(
            self.inverse(p_cov + eps)
        ) @ self.transpose(y_p_cov)

        if self.use_log_trace:
            rmi = 0.5 * self.log_trace(m + eps)
        else:
            rmi = 0.5 * self.log_det(m + eps)

        rmi = rmi / float(vector_size)

        return rmi.sum(dim=1).mean(dim=0)

    def extract_region_vector(self, x):
        x = self.downsample(x)
        stride = self.stride if self.downsampling_method == "region-extraction" else 1

        x_regions = F.unfold(x, kernel_size=self.radius, stride=stride)
        x_regions = x_regions.view((*x.shape[:2], self.radius**2, -1))
        return x_regions

    def downsample(self, x):
        if self.stride == 1:
            return x

        if self.downsampling_method == "region-extraction":
            return x

        padding = self.stride // 2
        if self.downsampling_method == "max":
            return F.max_pool2d(
                x, kernel_size=self.stride, stride=self.stride, padding=padding
            )
        if self.downsampling_method == "avg":
            return F.avg_pool2d(
                x, kernel_size=self.stride, stride=self.stride, padding=padding
            )
        raise ValueError(self.downsampling_method)

    @staticmethod
    def transpose(x):
        return x.transpose(-2, -1)

    @staticmethod
    def inverse(x):
        return torch.inverse(x)

    @staticmethod
    def log_trace(x):
        x = torch.linalg.cholesky(x)
        diag = torch.diagonal(x, dim1=-2, dim2=-1)
        return 2 * torch.sum(torch.log(diag + 1e-8), dim=-1)

    @staticmethod
    def log_det(x):
        return torch.logdet(x)


class CharbonnierLoss(torch.nn.Module):
    def __init__(self, eps=1e-3) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, img1, img2) -> Tensor:
        return elementwise_charbonnier_loss(img1, img2, eps=self.eps).mean()


class HybridSSIMRMIFuse(nn.Module):
    def __init__(self, weight_ratio=(1.0, 1.0), ssim_channel=1):
        super().__init__()
        # self.l1 = nn.L1Loss()
        self.ssim = SSIMLoss(channel=ssim_channel)
        self.rmi = RMILoss(bce_weight=0.6)
        self.weight_ratio = weight_ratio

    def forward(self, fuse, x):
        fuse = fuse.clip(0, 1)

        vis = x[:, 0:1]
        ir = x[:, 1:]

        ssim_loss = self.ssim(fuse, vis) + self.ssim(fuse, ir)
        rmi_loss = self.rmi(fuse, vis) + self.rmi(fuse, ir)

        loss_d = dict(ssim=ssim_loss, rmi=rmi_loss)

        loss = self.weight_ratio[0] * ssim_loss + self.weight_ratio[1] * rmi_loss
        return loss, loss_d


class HybridPIALoss(nn.Module):
    def __init__(self, weight_ratio=(3, 7, 20, 10)) -> None:
        super().__init__()
        assert (
            len(weight_ratio) == 4
        ), "@weight_ratio must be a tuple or list of length 4"
        self.weight_ratio = weight_ratio
        self._mcg_loss = MaxGradientLoss()
        self.perceptual_loss = PerceptualLoss(norm=True)

    def forward(self, fuse, gt):
        vis = gt[:, 0:1]
        ir = gt[:, 1:]

        l1_int = (F.l1_loss(fuse, vis) + F.l1_loss(fuse, ir)) * self.weight_ratio[0]
        l1_aux = (F.l1_loss(fuse, gt.max(1, keepdim=True)[0])) * self.weight_ratio[1]

        # FIXME: this should implement as the largest gradient of vis and ir
        # l1_grad = (F.l1_loss(gradient(fuse), gradient(vis)) + F.l1_loss(gradient(fuse), gradient(ir))) * \
        #           self.weight_ratio[2]
        l1_grad = self._mcg_loss(fuse, ir, vis) * self.weight_ratio[2]
        percep_loss = (
            self.perceptual_loss(fuse, vis) + self.perceptual_loss(fuse, ir)
        ) * self.weight_ratio[3]

        loss_d = dict(
            intensity_loss=l1_int,
            context_loss=l1_aux,
            gradient_loss=l1_grad,
            percep_loss=percep_loss,
        )

        return l1_int + l1_aux + l1_grad + percep_loss, loss_d


def parse_fusion_gt(gt: "Tensor | tuple[Tensor] | list[Tensor]"):
    # TODO: consider the vis is RGB
    if isinstance(gt, Tensor):
        if gt.size(1) == 4:
            ir, vi = gt[:, 3:], gt[:, :3]
        elif gt.size(1) == 2:
            ir, vi = gt[:, 1:], gt[:, 0:1]
    elif isinstance(gt, (tuple, list)):
        ir, vi = gt[1], gt[0]
    else:
        raise ValueError("gt must be a tensor or a tuple or a list")

    return ir, vi


# U2Fusion dynamic loss weight
class U2FusionLoss(nn.Module):
    def __init__(
        self, loss_weights: tuple[float, float, float] = (5.0, 2.0, 10.0)
    ) -> None:
        # loss_weights:
        super().__init__()
        # modified from https://github.com/ytZhang99/U2Fusion-pytorch/blob/master/train.py
        # and https://github.com/linklist2/PIAFusion_pytorch/blob/master/train_fusion_model.py
        # no normalization
        # so do not unormalize the input

        assert len(loss_weights) == 3, "loss_weights must be a tuple of length 3"

        self.feature_model = vgg16(pretrained=True)
        self.c = 0.1
        self.loss_weights = loss_weights
        self.ssim_loss = SSIMLoss(channel=1)
        #   , size_average=False)
        self.mse_loss = nn.MSELoss(reduction="none")

    def forward(self, fuse, gt, *, mask=None):
        # similiar to PIAFusion paper, which introduces a classifier
        # to judge the day or night image and give the probability
        ir, vi = parse_fusion_gt(gt)
        ir, vi = self.repeat_dims(ir), self.repeat_dims(vi)

        ws = self.dynamic_weight(ir, vi)
        ir_w, vi_w = ws.chunk(2, dim=-1)
        ir_w, vi_w = ir_w.flatten(), vi_w.flatten()

        # here we do not follow U2Fusion paper and change it into other losses
        l1_int = (
            vi_w * self.mse_loss(fuse, vi).mean((1, 2, 3))
            + ir_w * self.mse_loss(fuse, ir).mean((1, 2, 3))
        ).mean() * self.loss_weights[0]

        l1_aux = F.mse_loss(fuse, torch.max(ir, vi)) * self.loss_weights[1]

        # gradient part. choose the largest gradient
        # l1_grad = (
        #         F.l1_loss(
        #             gradient(fuse),
        #             torch.maximum(
        #                 vi_w[:, None, None, None] * gradient(gt[:, 0:1]),
        #                 ir_w[:, None, None, None] * gradient(gt[:, 1:]),
        #             ),
        #         )
        #         * self.loss_weights[2]
        # )

        # l1_grad = (
        #     self.l1_loss(gradient(fuse), vi_w * gradient(gt[:, 0:1])).mean((1, 2, 3))
        #     + self.l1_loss(gradient(fuse), ir_w * gradient(gt[:, 1:])).mean((1, 2, 3))
        # ).mean()

        # ssim loss would cause window artifacts
        # loss_ssim = (
        #     ir_w * self.ssim_loss(fuse, gt[:, 1:])
        #     + vi_w * self.ssim_loss(fuse, gt[:, 0:1])
        # ).mean() * self.loss_weights[2]
        loss_ssim = (
            self.ssim_loss(fuse, ir) + self.ssim_loss(fuse, vi)
        ) * self.loss_weights[2]

        loss_d = dict(intensity_loss=l1_int, aux_loss=l1_aux, ssim_loss=loss_ssim)
        # print(ir_w, vi_w)

        return l1_int + l1_aux + loss_ssim, loss_d

    @torch.no_grad()
    def dynamic_weight(self, ir_vgg, vi_vgg):
        ir_f = self.feature_model(ir_vgg)
        vi_f = self.feature_model(vi_vgg)

        m1s = []
        m2s = []
        for i in range(len(ir_f)):
            m1 = torch.mean(self.features_grad(ir_f[i]).pow(2), dim=[1, 2, 3])
            m2 = torch.mean(self.features_grad(vi_f[i]).pow(2), dim=[1, 2, 3])

            m1s.append(m1)
            m2s.append(m2)
            # if i == 0:
            #     w1 = torch.unsqueeze(m1, dim=-1)
            #     w2 = torch.unsqueeze(m2, dim=-1)
            # else:
            #     w1 = torch.cat((w1, torch.unsqueeze(m1, dim=-1)), dim=-1)
            #     w2 = torch.cat((w2, torch.unsqueeze(m2, dim=-1)), dim=-1)

        w1 = torch.stack(m1s, dim=-1)
        w2 = torch.stack(m2s, dim=-1)

        weight_1 = (torch.mean(w1, dim=-1) / self.c).detach()
        weight_2 = (torch.mean(w2, dim=-1) / self.c).detach()

        # print(weight_1.tolist()[:6], weight_2.tolist()[:6])

        weight_list = torch.stack(
            [weight_1, weight_2], dim=-1
        )  # torch.cat((weight_1.unsqueeze(-1), weight_2.unsqueeze(-1)), -1)
        weight_list = F.softmax(weight_list, dim=-1)

        return weight_list

    @staticmethod
    def features_grad(features):
        kernel = [[1 / 8, 1 / 8, 1 / 8], [1 / 8, -1, 1 / 8], [1 / 8, 1 / 8, 1 / 8]]
        kernel = (
            torch.FloatTensor(kernel)
            .expand(features.shape[1], 1, 3, 3)
            .to(features.device)
        )
        feat_grads = F.conv2d(
            features, kernel, stride=1, padding=1, groups=features.shape[1]
        )
        # _, c, _, _ = features.shape
        # c = int(c)
        # for i in range(c):
        #     feat_grad = F.conv2d(
        #         features[:, i : i + 1, :, :], kernel, stride=1, padding=1
        #     )
        #     if i == 0:
        #         feat_grads = feat_grad
        #     else:
        #         feat_grads = torch.cat((feat_grads, feat_grad), dim=1)
        return feat_grads

    def repeat_dims(self, x):
        assert x.size(1) in [1, 3], "the number of channel of x must be 3"
        if x.size(1) == 1:
            return x.repeat(1, 3, 1, 1)
        else:
            return x


# DCT Blur Loss
class DCTBlurLoss(nn.Module):
    def __init__(self, temperature=100, reduction="mean") -> None:
        super().__init__()
        self.t = temperature
        self.reduction = reduction
        self.distance = nn.L1Loss(reduction=reduction)
        if reduction == "none":
            self.feature_model = vgg16(pretrained=True)
            self.c = 0.1

    @staticmethod
    def heat_blur_torch(img, t=25):
        K1 = img.shape[-2]
        K2 = img.shape[-1]

        dct_img = dct_2d(img, norm="ortho")  # [3, K1, K2]
        freqs_h = torch.pi * torch.linspace(0, K1 - 1, K1) / K1  # [K1]
        freqs_w = torch.pi * torch.linspace(0, K2 - 1, K2) / K2  # [K2]

        freq_square = (freqs_h[:, None] ** 2 + freqs_w[None, :] ** 2).to(
            img.device
        )  # [K1, K2]
        dct_img = dct_img * torch.exp(-freq_square[None, ...] * t)  # [3, K1, K2]

        recon_img = idct_2d(dct_img, norm="ortho")

        return recon_img

    def forward(self, f, gt):
        ws = self.dynamic_weight(gt)
        ir_w, vi_w = ws.chunk(2, dim=-1)
        ir_w, vi_w = ir_w.flatten(), vi_w.flatten()

        f_dct_blur = self.heat_blur_torch(f, t=self.t)
        vi_dct_blur, ir_dct_blur = self.heat_blur_torch(gt, t=self.t).chunk(2, dim=1)

        f_vi_loss = self.distance(f_dct_blur, vi_dct_blur)
        f_ir_loss = self.distance(f_dct_blur, ir_dct_blur)

        if self.reduction == "none":
            f_vi_loss = f_vi_loss.mean(dim=(1, 2, 3))
            f_ir_loss = f_ir_loss.mean(dim=(1, 2, 3))

            ws = self.dynamic_weight(gt)
            ir_w, vi_w = ws.chunk(2, dim=-1)
            ir_w, vi_w = ir_w.flatten(), vi_w.flatten()

            f_vi_loss = f_vi_loss * vi_w
            f_ir_loss = f_ir_loss * ir_w

        return (f_vi_loss + f_ir_loss).mean()

    @torch.no_grad()
    def dynamic_weight(self, gt):
        ir_vgg, vi_vgg = self.repeat_dims(gt[:, 1:]), self.repeat_dims(gt[:, 0:1])

        ir_f = self.feature_model(ir_vgg)
        vi_f = self.feature_model(vi_vgg)

        m1s = []
        m2s = []
        for i in range(len(ir_f)):
            m1 = torch.mean(self.features_grad(ir_f[i]).pow(2), dim=[1, 2, 3])
            m2 = torch.mean(self.features_grad(vi_f[i]).pow(2), dim=[1, 2, 3])

            m1s.append(m1)
            m2s.append(m2)

        w1 = torch.stack(m1s, dim=-1)
        w2 = torch.stack(m2s, dim=-1)

        weight_1 = torch.mean(w1, dim=-1) / self.c
        weight_2 = torch.mean(w2, dim=-1) / self.c
        weight_list = torch.stack([weight_1, weight_2], dim=-1)
        weight_list = F.softmax(weight_list, dim=-1)

        return weight_list

    @staticmethod
    def features_grad(features):
        kernel = [[1 / 8, 1 / 8, 1 / 8], [1 / 8, -1, 1 / 8], [1 / 8, 1 / 8, 1 / 8]]
        kernel = (
            torch.FloatTensor(kernel)
            .expand(features.shape[1], 1, 3, 3)
            .to(features.device)
        )
        feat_grads = F.conv2d(
            features, kernel, stride=1, padding=1, groups=features.shape[1]
        )
        return feat_grads

    def repeat_dims(self, x):
        assert x.size(1) == 1, "the number of channel of x must be 1"
        return x.repeat(1, 3, 1, 1)


################# SwinFusion loss helper functions #################


class L_color(nn.Module):
    def __init__(self):
        super(L_color, self).__init__()

    def forward(self, x):
        b, c, h, w = x.shape

        mean_rgb = torch.mean(x, [2, 3], keepdim=True)
        mr, mg, mb = torch.split(mean_rgb, 1, dim=1)
        Drg = torch.pow(mr - mg, 2)
        Drb = torch.pow(mr - mb, 2)
        Dgb = torch.pow(mb - mg, 2)
        k = torch.pow(torch.pow(Drg, 2) + torch.pow(Drb, 2) + torch.pow(Dgb, 2), 0.5)
        return k


class L_Grad(nn.Module):
    def __init__(self):
        super(L_Grad, self).__init__()
        self.sobelconv = Sobelxy()

    def forward(self, image_A, image_B, image_fused):
        image_A_Y = image_A[:, :1, :, :]
        image_B_Y = image_B[:, :1, :, :]
        image_fused_Y = image_fused[:, :1, :, :]
        gradient_A = self.sobelconv(image_A_Y)
        # gradient_A = TF.gaussian_blur(gradient_A, 3, [1, 1])
        gradient_B = self.sobelconv(image_B_Y)
        # gradient_B = TF.gaussian_blur(gradient_B, 3, [1, 1])
        gradient_fused = self.sobelconv(image_fused_Y)
        gradient_joint = torch.max(gradient_A, gradient_B)
        Loss_gradient = F.l1_loss(gradient_fused, gradient_joint)
        return Loss_gradient


class Sobelxy(nn.Module):
    def __init__(self):
        super(Sobelxy, self).__init__()
        kernelx = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
        kernely = [[1, 2, 1], [0, 0, 0], [-1, -2, -1]]
        kernelx = torch.FloatTensor(kernelx).unsqueeze(0).unsqueeze(0)
        kernely = torch.FloatTensor(kernely).unsqueeze(0).unsqueeze(0)
        self.weightx = nn.Parameter(data=kernelx, requires_grad=False).cuda()
        self.weighty = nn.Parameter(data=kernely, requires_grad=False).cuda()

    def forward(self, x):
        sobelx = F.conv2d(x, self.weightx, padding=1)
        sobely = F.conv2d(x, self.weighty, padding=1)
        return torch.abs(sobelx) + torch.abs(sobely)


class L_SSIM(nn.Module):
    def __init__(self):
        super(L_SSIM, self).__init__()
        self.sobelconv = Sobelxy()

    def forward(self, image_A, image_B, image_fused):
        weight_A = 0.5
        weight_B = 0.5
        Loss_SSIM = weight_A * ssim(image_A, image_fused) + weight_B * ssim(
            image_B, image_fused
        )
        return Loss_SSIM


class L_Intensity(nn.Module):
    def __init__(self):
        super(L_Intensity, self).__init__()

    def forward(self, image_A, image_B, image_fused):
        image_A = image_A.unsqueeze(0)
        image_B = image_B.unsqueeze(0)
        intensity_joint = torch.mean(torch.cat([image_A, image_B]), dim=0)
        Loss_intensity = F.l1_loss(image_fused, intensity_joint)
        return Loss_intensity


################# SwinFusion loss #################


class SwinFusionLoss(nn.Module):
    def __init__(self):
        super(SwinFusionLoss, self).__init__()
        self.L_Grad = L_Grad()
        self.L_Inten = L_Intensity()
        self.L_SSIM = L_SSIM()

    def forward(self, image_A, image_B, image_fused):
        loss_l1 = 20 * self.L_Inten(image_A, image_B, image_fused)
        loss_gradient = 20 * self.L_Grad(image_A, image_B, image_fused)
        loss_SSIM = 10 * (1 - self.L_SSIM(image_A, image_B, image_fused))
        fusion_loss = loss_l1 + loss_gradient + loss_SSIM

        loss_d = {
            "loss_l1": loss_l1,
            "loss_gradient": loss_gradient,
            "loss_SSIM": loss_SSIM,
        }
        return fusion_loss, loss_d  # loss_gradient, loss_l1, loss_SSIM


#####################################################


class CDDFusionLoss(nn.Module):
    def __init__(self, weights=(1, 1, 1)) -> None:
        super().__init__()
        self.weights = weights

        self.l1_loss = nn.L1Loss()
        self.dct_loss = DCTBlurLoss(reduction="none")
        self.mcg_loss = MaxGradientLoss()

    def forward(self, f, gt):
        l1_loss = self.l1_loss(f, gt.max(dim=1, keepdim=True)[0]) * self.weights[0]
        dct_loss = self.dct_loss(f, gt) * self.weights[1]
        mcg_loss = self.mcg_loss(f, gt[:, 0:1], gt[:, 1:]) * self.weights[2]

        loss_d = dict(l1_loss=l1_loss, dct_loss=dct_loss, mcg_loss=mcg_loss)

        return l1_loss + dct_loss + mcg_loss, loss_d


### psfusion loss


class CorrelationLoss(nn.Module):
    """
    global normalized cross correlation (sqrt)
    """

    def __init__(self, eps=1e-6):
        super(CorrelationLoss, self).__init__()
        self.eps = eps

    def corr2(self, img1, img2):
        img1 = img1 - img1.mean()
        img2 = img2 - img2.mean()
        r = torch.sum(img1 * img2) / torch.sqrt(
            torch.sum(img1 * img1) * torch.sum(img2 * img2)
        )
        return r

    def forward(self, image_ir, img_vis, img_fusion):
        cc = self.corr2(image_ir, img_fusion) + self.corr2(img_vis, img_fusion)

        return 1.0 / (cc + self.eps)


# * ==========================================================
# * DRMF fusion loss
# * ==========================================================


def RGB2YCrCb(rgb_image):
    """
    Convert RGB format to YCrCb format.
    Used in the intermediate results of the color space conversion, because the default size of rgb_image is [B, C, H, W].
    :param rgb_image: image data in RGB format
    :return: Y, Cr, Cb
    """

    R = rgb_image[:, 0:1]
    G = rgb_image[:, 1:2]
    B = rgb_image[:, 2:3]
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    Cr = (R - Y) * 0.713 + 0.5
    Cb = (B - Y) * 0.564 + 0.5

    Y = Y.clamp(0.0, 1.0)
    Cr = Cr.clamp(0.0, 1.0)  # .detach()
    Cb = Cb.clamp(0.0, 1.0)  # .detach()
    return Y, Cb, Cr


def YCbCr2RGB(Y, Cb, Cr):
    """
    Convert YcrCb format to RGB format
    :param Y.
    :param Cb.
    :param Cr.
    :return.
    """
    ycrcb = torch.cat([Y, Cr, Cb], dim=1)
    B, C, W, H = ycrcb.shape
    im_flat = ycrcb.transpose(1, 3).transpose(1, 2).reshape(-1, 3)
    mat = torch.tensor(
        [[1.0, 1.0, 1.0], [1.403, -0.714, 0.0], [0.0, -0.344, 1.773]]
    ).to(Y.device)
    bias = torch.tensor([0.0 / 255, -0.5, -0.5]).to(Y.device)
    temp = (im_flat + bias).mm(mat)
    out = temp.reshape(B, W, H, C).transpose(1, 3).transpose(2, 3)
    out = out.clamp(0, 1.0)
    return out


class SobelOp(nn.Module):
    def __init__(self, in_c=3, out_c=3, mode="add"):
        super(SobelOp, self).__init__()
        kernelx = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
        kernely = [[1, 2, 1], [0, 0, 0], [-1, -2, -1]]
        kernelx = torch.tensor(kernelx, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        kernely = torch.tensor(kernely, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        kernelx = kernelx.repeat(out_c, 1, 1, 1)
        kernely = kernely.repeat(out_c, 1, 1, 1)
        self.register_buffer("weightx", kernelx)
        self.register_buffer("weighty", kernely)

        self.mode = mode
        assert mode in ["add", "max"], "mode should be add or max"

    def forward(self, x):
        sobelx = F.conv2d(x, self.weightx, padding=1, groups=x.size(1))
        sobely = F.conv2d(x, self.weighty, padding=1, groups=x.size(1))

        if self.mode == "add":
            sobel_xy = torch.abs(sobelx) + torch.abs(sobely)
        elif self.mode == "max":
            sobel_xy = torch.max(torch.abs(sobelx), torch.abs(sobely))
        else:
            raise ValueError("mode should be add or max")

        return sobel_xy


class DRMFFusionLoss(nn.Module):
    def __init__(
        self,
        *,
        latent_weighted: bool = False,
        grad_loss: bool = True,
        ssim_loss: bool = True,
        tv_loss: bool = False,
        pseudo_l1_const=0.0,
        boundary_loss: bool = True,
        correlation_loss: bool = False,
        mask_loss: bool = False,
        reduce_label: bool = True,
        color_loss: bool = True,
        color_loss_bg_masked: bool = False,
        lpips_loss: bool = False,
        grad_op: str = "sobel_add",
        grad_only_on_Y: bool = True,
        still_boundary_loss_when_gt: bool = True,
        prior: str = "mean",
        weight_dict: dict = None,
        grad_norm: bool = False,
        ssim_implm_by: str = "kornia",
        ssim_window_size: int = 11,
    ):
        """
        Initialize the DRMFFusionLoss module.

        Args:
            latent_weighted (bool): Whether to use latent weighting. Default is False.
            grad_loss (bool): Whether to include gradient loss. Default is True.
            ssim_loss (bool): Whether to include SSIM loss. Default is True.
            tv_loss (bool): Whether to include total variation loss. Default is False.
            pseudo_l1_const (float): Constant for pseudo L1 loss. Default is 0.
            boundary_loss (bool): Whether to include boundary loss. Default is True.
            correlation_loss (bool): Whether to include correlation loss. Default is False.
            mask_loss (bool): Whether to include mask loss. Default is False.
            reduce_label (bool): Whether to reduce labels. Default is True.
            color_loss (bool): Whether to include color loss. Default is True.
            color_loss_bg_masked (bool): Whether to mask background in color loss. Default is False.
            lpips_loss (bool): Whether to include LPIPS loss. Default is False.
            grad_op (str): Gradient operation to use. Default is 'sobel_add'.
            grad_only_on_Y (bool): Whether to apply gradient only on Y channel. Default is True.
            still_boundary_loss_when_gt (bool): Whether to still use boundary loss when ground truth is available. Default is True.
            prior (str): Prior to use. Default is 'mean'.
            weight_dict (dict): Dictionary of loss weights. Default is None.
            grad_norm (bool): Whether to normalize gradients. Default is False.
            ssim_implm_by (str): SSIM implementation to use. Default is 'kornia'.
            ssim_window_size (int): Window size for SSIM calculation. Default is 11.

        Notes:
            In FILM (https://arxiv.org/pdf/2402.02235), the loss weights are set as follows:
                `\\mathcal L_{total} = \\mathcal L_{inten} + {300/500} \\mathcal L_{grad} + 1 \times \\mathcal L_{ssim}` in MEF, MFF tasks.
            but in VIF tasks, the weights are set as follows:
                `\\mathcal L_{total} = \\mathcal L_{inten} + 40 \\mathcal L_{grad} + 0 \times \\mathcal L_{ssim}`
        """
        super(DRMFFusionLoss, self).__init__()
        self.latent_weighted = latent_weighted
        self.use_prior = not is_none(prior)
        self.boundary = boundary_loss
        self.ssim = ssim_loss
        self.grad = grad_loss
        self.lpips = lpips_loss
        self.tv = tv_loss
        self.correlation = correlation_loss
        self.reduce_label = reduce_label
        self.color_loss = color_loss
        self.mask_loss = mask_loss
        self.color_loss_bg_masked = color_loss_bg_masked
        self.still_boundary_loss_when_gt = still_boundary_loss_when_gt
        self.prior = prior
        self.grad_only_on_Y = grad_only_on_Y
        self.grad_norm = grad_norm
        if mask_loss:
            logger.info(
                f'{__class__.__name__}: mask loss performs [green]{"only on background"
                         if color_loss_bg_masked else "on the whole image"}[/green]'
            )

        self.loss_func = (
            nn.L1Loss(reduction="none")
            if pseudo_l1_const == 0
            else partial(self.pseudo_l2_loss, c=pseudo_l1_const)
        )
        main_loss = "l1 loss" if pseudo_l1_const == 0 else "pseudo l2 loss"

        assert (
            self.boundary or self.use_prior
        ), "one or both of boundary loss and prior ([max or mean]) should be used"

        if grad_loss:
            if grad_op.startswith("sobel"):
                grad_op, mode = grad_op.split("_")
                in_c = out_c = 1 if grad_only_on_Y else 3
                self.grad_op = SobelOp(in_c, out_c, mode)
            elif grad_op.startswith("k_sobel"):
                order = int(grad_op.split("_")[-1])
                assert order in [1, 2], "order should be 1 or 2"
                self.grad_op = lambda img: spatial_gradient(
                    img, mode="sobel", order=order, normalized=False
                ).max(dim=2)[0]
            elif grad_op.startswith("k_laplacian"):
                k = int(grad_op.split("_")[-1])
                # kernel size recommand to be 7 or 11
                assert (
                    k <= 13
                ), "kernel size should be less than 13, larger kernel size brings higher computational cost"
                self.grad_op = lambda img: laplacian(
                    img, kernel_size=k, normalized=False
                )
            else:
                raise ValueError(
                    "grad_op should be sobel or starts with k_sobel or k_laplacian"
                )
        if ssim_loss:
            assert self.grad, "ssim loss should be used with grad loss"
            self.ssim_func = get_ssim_loss(
                implem_by=ssim_implm_by,
                window_size=ssim_window_size,
                channel=1 if grad_only_on_Y else 3,
            )
        if tv_loss:
            from deepinv.loss import TVLoss

            self.tv_loss = TVLoss(weight=2.0)
        if correlation_loss:
            self.cc_loss = CorrelationLoss()
        if lpips_loss:
            self.lpips_loss = PerceptualLoss(
                norm=True
            )  # norm: image value range is changed to [-1, 1] from [0, 1]

        if latent_weighted:
            assert (
                self.boundary
            ), "if using latent weighted, boundary loss should be used"
            logger.info("using vgg16 to extract latent features")
            self.latent_model = vgg16(pretrained=True).eval()
            self.latent_temp = 0.1
            feature_grad_kernel = torch.tensor(
                [[1 / 8, 1 / 8, 1 / 8], [1 / 8, -1, 1 / 8], [1 / 8, 1 / 8, 1 / 8]]
            ).type(torch.float32)
            self.register_buffer("kernel", feature_grad_kernel)

        # rescale the loss using weight_dict
        self.weight_dict = default(
            weight_dict,
            {
                "fusion_gt": 10.0,
                "inten_f_joint": 1.0,  # prior used
                "inten_f_ir": 1.0,
                "inten_f_vi": 1.0,
                "color_f_cb": 2.0,
                "color_f_cr": 2.0,
                "grad_f_joint": 2,
                ## not used
                # 'grad_f_ir': 20,
                # 'grad_f_vi': 20,
                "ssim_f_joint": 0.6,
                "tv_f": 0.1,
                "crr_f": 0.02,
                "lpips_f_gt": 0.2,
                "lpips_f_ir": 0.2,
                "lpips_f_vi": 0.2,
                "lpips_f_joint": 0.2,
            },
        )
        # if normalizing gradient for each loss, we do not rescale the loss using `self.weight_dict`
        if grad_norm:
            for k in self.weight_dict.keys():
                self.weight_dict[k] = 1.0

        # print loss config
        logger.info(
            f"Latent weighted: {latent_weighted}, TV loss: {tv_loss}, ",
            f"grad loss: {grad_loss} with {grad_op}, SSIM loss: {ssim_loss}, color loss: {color_loss}, ",
            f"boundary loss: {boundary_loss}, ",
            f'{"use" if self.use_prior else "do not use"} prior [g]{self.prior}[/g], ',
            f"lpips loss: {lpips_loss}, correlation loss: {correlation_loss}, ",
            f"mask loss: {mask_loss}, ",
            f"grad_only_on_Y: {grad_only_on_Y}, ",
            f"grad_norm: {grad_norm} (balance losses or use weight dict)",
        )

        logger.info(f"main loss: {main_loss}")
        logger.info(f"weight dict: {self.weight_dict}")

    @staticmethod
    def pseudo_l2_loss(img1, img2, c):
        return torch.sqrt((img1 - img2) ** 2 + c**2) - c

    @staticmethod
    def check_rgb(img):
        if not_color_img := img.size(1) != 3:
            assert img.size(1) == 1, "The channel of the image should be 1 or 3."
            img = img.repeat(1, 3, 1, 1)

        return img.detach(), (not not_color_img)

    @torch.no_grad()
    def dynamic_weight(self, ir, vi):
        def features_grad(features):
            kernel = self.kernel.expand(features.shape[1], 1, 3, 3)
            feat_grads = F.conv2d(
                features, kernel, stride=1, padding=1, groups=features.shape[1]
            )
            return feat_grads

        ir_f = self.latent_model(ir)
        vi_f = self.latent_model(vi)

        m1s = []
        m2s = []
        for i in range(len(ir_f)):
            m1 = torch.mean(features_grad(ir_f[i]).pow(2), dim=[1, 2, 3])
            m2 = torch.mean(features_grad(vi_f[i]).pow(2), dim=[1, 2, 3])

            m1s.append(m1)
            m2s.append(m2)

        w1 = torch.stack(m1s, dim=-1)
        w2 = torch.stack(m2s, dim=-1)

        weight_1 = torch.mean(w1, dim=-1) / self.latent_temp
        weight_2 = torch.mean(w2, dim=-1) / self.latent_temp

        weight_list = torch.stack([weight_1, weight_2], dim=-1)
        weight_list = F.softmax(weight_list, dim=-1)
        ir_w, vi_w = weight_list.chunk(2, dim=-1)
        ir_w, vi_w = ir_w.flatten(), vi_w.flatten()

        return vi_w, ir_w

    def check_dtype_and_device(self, *args: tuple[Tensor]):
        dtype = None
        device = None

        def _asserts(ti, dtype, device):
            assert (
                ti.dtype == dtype
            ), f"The dtype of the input tensors should be the same, but got {ti.dtype} and {dtype}"
            assert (
                ti.device == device
            ), f"The device of the input tensors should be the same, but got {ti.device} and {device}"

        for t in args:
            if dtype is None:
                dtype = t.dtype
            if device is None:
                device = t.device

            if isinstance(t, (tuple, list)):
                for ti in t:
                    _asserts(ti, dtype, device)
            else:
                _asserts(t, dtype, device)

    def split_boundary_gt_tensor(self, boundary_gt: Tensor):
        assert boundary_gt.size(1) in [
            2,
            4,
            6,
        ], "The channel of the boundary_gt should be 2, 4, or 6."

        # 1 for vi, 1 for ir
        # or just Y component of RGB images
        if boundary_gt.size(1) == 2:
            ir, vi = boundary_gt[:, 1:], boundary_gt[:, :1]
        # 1 for ir, 3 for vi
        elif boundary_gt.size(1) == 4:
            ir, vi = boundary_gt[:, 3:], boundary_gt[:, :3]
        # 3 for over/far, 3 for under/near
        elif boundary_gt.size(1) == 6:
            ir, vi = boundary_gt[:, 3:], boundary_gt[:, :3]
        else:
            raise ValueError(
                f"The channel of the boundary_gt should be 2, 4, or 6, but got {boundary_gt.size(1)}"
            )

        return vi, ir

    def forward(
        self,
        img_fusion: Tensor,
        boundary_gt: "Tensor | tuple",  # cat([vi, ir]) or tuple(vi, ir)
        fusion_gt: "Tensor" = None,  # ground truth provided in dataset
        mask: "Tensor | None" = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if mask is not None and self.mask_loss:
            self.check_dtype_and_device(img_fusion, boundary_gt, mask)
            with torch.no_grad():
                if self.reduce_label:
                    mask2 = mask.detach().clone()
                    mask2[mask2 > 1.0] = 1.0
                else:
                    mask2 = mask

            ## TODO: consider this case
            # mask.size(1) == 2, two boundaries all have masks
            if mask2.size(1) == 2:
                assert False, "mask.size(1) == 2, two boundaries all have masks"
                mask_B, mask_A = mask2.chunk(2, dim=1)
            else:
                mask_B, mask_A = mask2, mask2
        elif not self.mask_loss:
            mask2 = None  # cast to None

        self.check_dtype_and_device(img_fusion, boundary_gt)

        wd = self.weight_dict
        loss_intensity = 0
        loss_color = 0
        loss_grad = 0
        loss_fusion = 0
        loss = {}

        # split boundary gt
        no_batch_ndim = img_fusion.ndim - 1
        broadcast_fn = lambda x: x.reshape(-1, *[1] * no_batch_ndim)  # noqa: py3.11 supported

        # img_A: ir/under/MRI, img_B: vi/over/spect
        if isinstance(boundary_gt, (tuple, list)):
            img_B, img_A = boundary_gt
        else:
            img_B, img_A = self.split_boundary_gt_tensor(boundary_gt)

        # if has gt (e.g., when we train model on multi-exposure image fusion task)
        if exist_gt := exists(fusion_gt):
            gt_loss = wd["fusion_gt"] * self.loss_func(img_fusion, fusion_gt).nanmean()
            loss["gt_loss"] = gt_loss
            loss_fusion += gt_loss

        if self.grad_norm:
            img_fusion = grad_norm(img_fusion)

        # ir, vi (detached in this function)
        (img_A, A_is_color), (img_B, B_is_color) = (
            self.check_rgb(img_A),
            self.check_rgb(img_B),
        )

        ## vgg latent weights
        # the latent weights comes from U2Fusion paper
        if self.latent_weighted:
            vi_w, ir_w = self.dynamic_weight(img_A, img_B)
            vi_w = broadcast_fn(vi_w)
            ir_w = broadcast_fn(ir_w)
        else:
            vi_w, ir_w = 1.0, 1.0

        # YCbCr decomposition
        # use for intensity, color, and gradient loss
        Y_fusion, Cb_fusion, Cr_fusion = kornia.color.rgb_to_ycbcr(img_fusion).chunk(
            3, dim=1
        )
        Y_A, Cb_A, Cr_A = kornia.color.rgb_to_ycbcr(img_A).chunk(3, dim=1)  # ir
        Y_B, Cb_B, Cr_B = kornia.color.rgb_to_ycbcr(img_B).chunk(3, dim=1)  # vi

        ## intensity and color loss
        # * compute boundary intensity and color loss when GT is not provided
        # * or we omit computing the loss
        # * in this case, we can say intensity and color are supervised by the GT
        # * we do not need to use the unsupervised loss
        # * `still_boundary_loss_when_gt` is to keep the boundary loss when GT is provided (optional)
        if (not exist_gt) or self.still_boundary_loss_when_gt:
            if self.prior == "max":
                Y_joint = torch.max(Y_A, Y_B)
            elif self.prior == "mean":
                Y_joint = (Y_A + Y_B) / 2

            if self.grad_norm:
                Y_fusion = grad_norm(Y_fusion)
                Cb_fusion = grad_norm(Cb_fusion)
                Cr_fusion = grad_norm(Cr_fusion)

            if mask2 is not None:
                # * intensity loss
                # 1. || fusion - max(vi, ir) ||                     max fusion
                # 2. || mask * fusion - mask * ir ||                ir loss (pedistrain)
                # 3. || (1 - mask) * fusion - (1 - mask) * vi ||    vi loss (background)

                loss_intensity = (
                    (
                        wd["inten_f_joint"] * self.loss_func(Y_fusion, Y_joint)
                        if self.use_prior
                        else 0.0
                    )
                    + (
                        wd["inten_f_ir"]
                        * ir_w
                        * self.loss_func(mask2 * Y_fusion, mask2 * Y_A)
                        if self.boundary
                        else 0.0
                    )
                    + (
                        wd["inten_f_vi"]
                        * vi_w
                        * self.loss_func(Y_fusion * (1 - mask2), Y_B * (1 - mask2))
                        if self.boundary
                        else 0.0
                    )
                )
                # * color loss
                # 1. || (1-mask) * fusion_Cb - (1-mask) * ir_Cb ||          ir_Cb loss
                # 2. || (1-mask) * fusion_Cr - (1-mask) * ir_Cr ||          ir_Cr loss
                if self.color_loss_bg_masked:
                    bg_mask = 1.0 - mask2
                else:
                    bg_mask = torch.ones_like(mask2)
                if self.color_loss:
                    loss_color = 0.0
                    if B_is_color:
                        loss_color += wd["color_f_cb"] * vi_w * self.loss_func(
                            Cb_fusion * bg_mask, Cb_B * bg_mask
                        ) + wd["color_f_cr"] * ir_w * self.loss_func(
                            Cr_fusion * bg_mask, Cr_B * bg_mask
                        )
                    if A_is_color:
                        loss_color += wd["color_f_cb"] * ir_w * self.loss_func(
                            Cb_fusion * bg_mask, Cb_A * bg_mask
                        ) + wd["color_f_cr"] * vi_w * self.loss_func(
                            Cr_fusion * bg_mask, Cr_A * bg_mask
                        )
            else:
                loss_intensity = (
                    (
                        wd["inten_f_joint"] * self.loss_func(Y_fusion, Y_joint)
                        if self.use_prior
                        else 0.0
                    )
                    + (
                        wd["inten_f_ir"] * ir_w * self.loss_func(Y_fusion, Y_A)
                        if self.boundary
                        else 0.0
                    )
                    + (
                        wd["inten_f_vi"] * vi_w * self.loss_func(Y_fusion, Y_B)
                        if self.boundary
                        else 0.0
                    )
                )

                if self.color_loss:
                    loss_color = 0.0
                    if B_is_color:
                        loss_color += wd["color_f_cb"] * self.loss_func(
                            Cb_fusion, Cb_B
                        ) + wd["color_f_cr"] * self.loss_func(Cr_fusion, Cr_B)
                    if A_is_color:
                        loss_color += wd["color_f_cb"] * self.loss_func(
                            Cb_fusion, Cb_A
                        ) + wd["color_f_cr"] * self.loss_func(Cr_fusion, Cr_A)

            loss_intensity = loss_intensity.nanmean()
            loss_fusion += loss_intensity
            loss["intensity_loss"] = loss_intensity
            if self.color_loss:
                loss_color = loss_color.nanmean()
                loss_fusion += loss_color
                loss["loss_color"] = loss_color

        ## lpips loss
        # * lpips loss is to enhance perceptual visuality
        if self.lpips:
            lpips_loss = 0.0
            if exist_gt:
                lpips_loss += self.lpips_loss(img_fusion, fusion_gt) * wd["lpips_f_gt"]

            # if self.still_boundary_loss_when_gt or (not exist_gt):
            #     lpips_loss += self.lpips_loss(img_fusion, img_A) * wd['lpips_f_ir'] + \
            #                   self.lpips_loss(img_fusion, img_B) * wd['lpips_f_vi']
            #     if self.use_prior:
            #         if self.prior == 'max':
            #             img_joint = torch.max(img_A, img_B)
            #         elif self.prior == 'mean':
            #             img_joint = (img_A + img_B) / 2
            #         lpips_loss += self.lpips_loss(img_fusion, img_joint) * wd['lpips_f_joint']

            loss_fusion += lpips_loss
            loss["lpips_loss"] = lpips_loss

        ## grad loss
        # * gradient loss is to enhance the fused image
        # * keep it although the GT is provided
        if self.grad:
            if self.grad_only_on_Y:
                grad_A = self.grad_op(Y_A)
                grad_B = self.grad_op(Y_B)
                grad_fusion = self.grad_op(Y_fusion)
            else:
                grad_A = self.grad_op(img_A)
                grad_B = self.grad_op(img_B)
                grad_fusion = self.grad_op(img_fusion)

            if self.grad_norm:
                grad_fusion = grad_norm(grad_fusion)

            grad_joint = torch.max(grad_A, grad_B)
            loss_grad += wd["grad_f_joint"] * self.loss_func(grad_fusion, grad_joint)
            # if mask is not None:
            #     mask_expand = mask
            #     if grad_fusion.ndim > 4:
            #         mask_expand = mask_expand.unsqueeze(1)
            #     loss_grad += wd['grad_f_ir'] * self.loss_func(grad_fusion * mask_expand, grad_A * mask_expand)

            # loss_grad += wd['grad_f_ir'] * self.loss_func(grad_fusion, grad_A) + \
            #              wd['grad_f_vi'] * self.loss_func(grad_fusion, grad_B)

            loss_grad = loss_grad.nanmean()

            loss_fusion += loss_grad
            loss.update({"loss_grad": loss_grad})

        ## ssim loss
        # * ssim loss is to enhance the fused image
        # * keep it although the GT is provided
        if self.ssim:
            w_A = grad_A.norm()
            w_B = grad_B.norm()
            Z = w_A + w_B
            w_A /= Z
            w_B /= Z

            if self.grad_only_on_Y:
                ssim_A = self.ssim_func(Y_fusion, Y_A)
                ssim_B = self.ssim_func(Y_fusion, Y_B)
            else:
                ssim_A = self.ssim_func(img_fusion, img_A)
                ssim_B = self.ssim_func(img_fusion, img_B)
            loss_ssim = wd["ssim_f_joint"] * (w_A * ssim_A + w_B * ssim_B)
            loss_fusion += loss_ssim
            loss.update({"loss_ssim": loss_ssim})

        ## tv loss
        if self.tv:
            img_fusion_tv = img_fusion
            if self.grad_norm:
                img_fusion_tv = grad_norm(img_fusion_tv)
            tv_loss = wd["tv_f"] * self.tv_loss(img_fusion_tv).nanmean()
            loss_fusion += tv_loss
            loss.update({"tv_loss": tv_loss})

        ## correlation loss
        if self.correlation:
            img_fusion_tv = img_fusion
            if self.grad_norm:
                img_fusion_tv = grad_norm(img_fusion_tv)
            loss_corr = wd["crr_f"] * self.cc_loss(img_A, img_B, img_fusion_tv)
            loss_fusion += loss_corr
            loss.update({"loss_corr": loss_corr})

        loss.update({"loss_fusion": loss_fusion})

        return loss_fusion, loss


## EMMA stage two fusion training loss


class EMMAFusionLoss(nn.Module):
    def __init__(
        self,
        fusion_model: "BaseModel",
        to_source_A_model: nn.Module,
        to_source_B_model: nn.Module,
        A_pretrain_path: str,
        B_pretrain_path: str,
        A_model_kwargs: dict,
        B_model_kwargs: dict,
        translation_kwargs: dict = {},
        main_once_fusion_loss: Union[callable, nn.Module] = None,
        refusion_weight: float = 0.1,
        detach_fused: bool = False,
        translation_weight: float = 1.0,
        model_pred_y: bool = True,
    ):
        super().__init__()
        device = next(fusion_model.parameters()).device
        # device = 'cuda:1'

        self.fusion_model = fusion_model

        self.A_model = to_source_A_model(**A_model_kwargs).to(device)
        self.A_model.load_state_dict(torch.load(A_pretrain_path))
        self.A_model.eval()
        logger.info(f"load A_model {self.A_model.__class__} done.")

        self.B_model = to_source_B_model(**B_model_kwargs).to(device)
        self.B_model.load_state_dict(torch.load(B_pretrain_path))
        self.B_model.eval()
        logger.info(f"load B_model {self.B_model.__class__} done.")

        self.shift_n = translation_kwargs.get("shift_num", 3)
        self.rotate_n = translation_kwargs.get("rotate_num", 3)
        self.flip_n = translation_kwargs.get("flip_num", 3)
        logger.info(f"translation params: {translation_kwargs}")
        logger.warning(
            f"{__class__.__name__}: notice that your batch size will be enlarged by setting shift_n, "
            + f"rotate_n, flip_n, total [red]x{self.shift_n + self.rotate_n + self.flip_n} times[/red]"
        )

        # apply some loss to first fusion image
        self.main_once_fusion_loss = main_once_fusion_loss

        self.refusion_weight = refusion_weight
        self.detach_fused = detach_fused
        self.translation_weight = translation_weight
        self.model_is_y_pred = model_pred_y

    def translation_loss(self, fused_img, s_A, s_B, mask=None):
        # once fused image by outter training loop
        # i.e., fusion_model(s_A, s_B)

        # fused image
        if fused_img.size(1) == 3:  # if input image is rgb
            y_cbcr = kornia.color.rgb_to_ycbcr(fused_img)
            Y, Cb, Cr = torch.split(y_cbcr, 1, dim=1)
            F_to_A = self.A_model(Y).clip(0, 1)
            F_to_B = self.B_model(Y).clip(0, 1)

            F_to_A = kornia.color.ycbcr_to_rgb(torch.cat([F_to_A, Cb, Cr], dim=1))
            # F_to_B = kornia.color.ycbcr_to_rgb(torch.cat([F_to_B, Cb, Cr], dim=1))
        else:  # if input image is gray image
            F_to_A = self.A_model(fused_img)
            F_to_B = self.B_model(fused_img)

        # translation fused image

        # NOTE: note that the implementation should double
        # the computation graph to calculate the refusion loss
        # this may cause the GPU memory issue.
        if self.detach_fused:
            fused_img_detach = fused_img.detach()
        else:
            fused_img_detach = fused_img
        trans_fused_img = self.apply_translation(fused_img_detach)

        if fused_img.size(1) == 3:  # if input image is rgb
            y_cbcr = kornia.color.rgb_to_ycbcr(trans_fused_img)
            Y, Cb, Cr = torch.split(y_cbcr, 1, dim=1)
            Ft_to_A = self.A_model(Y).clip(0, 1)
            Ft_to_B = self.B_model(Y).clip(0, 1)

            Ft_to_A = kornia.color.ycbcr_to_rgb(torch.cat([Ft_to_A, Cb, Cr], dim=1))
            # Ft_to_B = kornia.color.ycbcr_to_rgb(torch.cat([Ft_to_B, Cb, Cr], dim=1))
        else:  # if input image is gray image
            Ft_to_A = self.A_model(trans_fused_img)
            Ft_to_B = self.B_model(trans_fused_img)

        # refusion
        if self.model_is_y_pred:
            Ft_to_A_ycbcr = kornia.color.rgb_to_ycbcr(Ft_to_A)
            _Ft_A_y = Ft_to_A_ycbcr[:, :1]
            _Ft_A_cbcr = Ft_to_A_ycbcr[:, 1:]
        else:
            _Ft_A_y = Ft_to_A
            _Ft_A_cbcr = None
        Ft_refused = self.fusion_model.only_fusion_step(_Ft_A_y, Ft_to_B)
        if self.model_is_y_pred:
            Ft_refused = kornia.color.ycbcr_to_rgb(
                torch.cat([Ft_refused, _Ft_A_cbcr], dim=1)
            )

        # three losses
        # 1. source A loss
        loss_A = self.translation_basic_loss(F_to_A, s_A)

        # 2. source B loss
        loss_B = self.translation_basic_loss(F_to_B, s_B)

        # 3. refusion loss
        loss_refusion = (
            self.translation_basic_loss(Ft_refused, trans_fused_img)
            * self.refusion_weight
        )

        return loss_A + loss_B + loss_refusion

    def translation_basic_loss(self, Ft_to_any, source):
        l1_loss = F.l1_loss(Ft_to_any, source)
        grad_loss = F.l1_loss(spatial_gradient(Ft_to_any), spatial_gradient(source))

        return l1_loss + grad_loss

    def forward(
        self,
        fused: torch.Tensor,
        source_AB: "torch.Tensor | tuple[torch.Tensor, torch.Tensor]",
        mask: torch.Tensor = None,
    ):
        loss_dict = {}

        if isinstance(source_AB, Sequence):
            A, B = source_AB
        elif isinstance(source_AB, torch.Tensor):
            A, B = source_AB[:, :3], source_AB[:, 3:]
        else:
            raise ValueError("source_AB should be a tuple or a tensor")

        if self.main_once_fusion_loss:
            fused_loss, _ = self.main_once_fusion_loss(fused, (A, B), mask=mask)
            loss_dict["fusion_loss"] = fused_loss

        translation_loss = self.translation_loss(fused, A, B) * self.translation_weight
        loss_dict["tra_loss"] = translation_loss

        return fused_loss + translation_loss, loss_dict

    def apply_translation(self, x):
        if self.shift_n > 0:
            x_shift = self.shift_random(x, self.shift_n)
        if self.rotate_n > 0:
            x_rotate = self.rotate_random(x, self.rotate_n)
        if self.flip_n > 0:
            x_flip = self.flip_random(x, self.flip_n)

        if self.shift_n > 0:
            x = torch.cat((x, x_shift), 0)
        if self.rotate_n > 0:
            x = torch.cat((x, x_rotate), 0)
        if self.flip_n > 0:
            x = torch.cat((x, x_flip), 0)

        return x

    @staticmethod
    def shift_random(x, n_trans=5):
        H, W = x.shape[-2], x.shape[-1]
        assert (
            n_trans <= H - 1 and n_trans <= W - 1
        ), "n_shifts should less than {}".format(H - 1)
        shifts_row = random.sample(
            list(np.concatenate([-1 * np.arange(1, H), np.arange(1, H)])), n_trans
        )
        shifts_col = random.sample(
            list(np.concatenate([-1 * np.arange(1, W), np.arange(1, W)])), n_trans
        )
        x = torch.cat(
            [
                torch.roll(x, shifts=[sx, sy], dims=[-2, -1]).type_as(x)
                for sx, sy in zip(shifts_row, shifts_col)
            ],
            dim=0,
        )

        return x

    @staticmethod
    def rotate_random(data, n_trans=5, random_rotate=False):
        if random_rotate:
            theta_list = random.sample(list(np.arange(1, 359)), n_trans)
        else:
            theta_list = np.arange(10, 360, int(360 / n_trans))
        # data = torch.cat([kornia.geometry.rotate(data, torch.Tensor([theta]).type_as(data))for theta in theta_list], dim=0)
        d = []
        for theta in theta_list:
            d.append(kornia.geometry.rotate(data, torch.tensor(theta).to(data)))

        return torch.cat(d, dim=0)

    @staticmethod
    def flip_random(data, n_trans=3):
        assert n_trans <= 3, "n_flip should less than 3"

        if n_trans >= 1:
            data1 = kornia.geometry.transform.hflip(data)
        if n_trans >= 2:
            data2 = kornia.geometry.transform.vflip(data)
            data1 = torch.cat((data1, data2), 0)
        if n_trans == 3:
            data1 = torch.cat((data1, kornia.geometry.transform.hflip(data2)), 0)

        return data1


# ============================== FILM loss ==============================


class LpLssimLossweight(nn.Module):
    def __init__(self, window_size=5, size_average=True):
        """
        Constructor
        """
        super().__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = self.create_window(window_size, self.channel)

    def gaussian(self, window_size, sigma):
        """
        Get the gaussian kernel which will be used in SSIM computation
        """
        gauss = torch.Tensor(
            [
                exp(-((x - window_size // 2) ** 2) / float(2 * sigma**2))
                for x in range(window_size)
            ]
        )
        return gauss / gauss.sum()

    def create_window(self, window_size, channel):
        """
        Create the gaussian window
        """
        _1D_window = self.gaussian(window_size, 1.5).unsqueeze(1)  # [window_size, 1]
        _2D_window = (
            _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        )  # [1,1,window_size, window_size]
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window

    def _ssim(self, img1, img2, window, window_size, channel, size_average=True):
        """
        Compute the SSIM for the given two image
        The original source is here: https://stackoverflow.com/questions/39051451/ssim-ms-ssim-for-tensorflow
        """
        mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = (
            F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel)
            - mu1_sq
        )
        sigma2_sq = (
            F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel)
            - mu2_sq
        )
        sigma12 = (
            F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel)
            - mu1_mu2
        )

        C1 = 0.01**2
        C2 = 0.03**2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
            (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
        )

        if size_average:
            return ssim_map.mean()
        else:
            return ssim_map.mean(1).mean(1).mean(1)

    def forward(self, image_in, image_out, weight):
        # Check if need to create the gaussian window
        (_, channel, _, _) = image_in.size()
        if channel == self.channel and self.window.data.type() == image_in.data.type():
            pass
        else:
            window = self.create_window(self.window_size, channel)
            window = window.to(image_out.get_device())
            window = window.type_as(image_in)
            self.window = window
            self.channel = channel

        # Lp
        Lp = torch.sqrt(torch.sum(torch.pow((image_in - image_out), 2)))  # 二范数
        # Lp = torch.sum(torch.abs(image_in - image_out))  # 一范数
        # Lssim
        Lssim = 1 - self._ssim(
            image_in,
            image_out,
            self.window,
            self.window_size,
            self.channel,
            self.size_average,
        )
        return Lp + Lssim * weight, Lp, Lssim * weight


class Fusionloss(nn.Module):
    def __init__(self, coeff_int=1, coeff_grad=10, in_max=True, device="cuda"):
        super(Fusionloss, self).__init__()
        self.sobelconv = Sobelxy(device=device)
        self.coeff_int = coeff_int
        self.coeff_grad = coeff_grad
        self.in_max = in_max

    def forward(self, image_vis, image_ir, generate_img):
        image_y = image_vis[:, :1, :, :]
        if self.in_max:
            x_in_max = torch.max(image_y, image_ir)
        else:
            x_in_max = (image_y + image_ir) / 2.0
        loss_in = F.l1_loss(x_in_max, generate_img)
        y_grad = self.sobelconv(image_y)
        ir_grad = self.sobelconv(image_ir)
        generate_img_grad = self.sobelconv(generate_img)
        x_grad_joint = torch.max(y_grad, ir_grad)
        loss_grad = F.l1_loss(x_grad_joint, generate_img_grad)
        loss_total = self.coeff_int * loss_in + self.coeff_grad * loss_grad
        return loss_total, loss_in, loss_grad


class Sobelxy(nn.Module):
    def __init__(self, device="cuda"):
        super(Sobelxy, self).__init__()
        kernelx = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
        kernely = [[1, 2, 1], [0, 0, 0], [-1, -2, -1]]
        kernelx = torch.FloatTensor(kernelx).unsqueeze(0).unsqueeze(0)
        kernely = torch.FloatTensor(kernely).unsqueeze(0).unsqueeze(0)
        self.weightx = nn.Parameter(data=kernelx, requires_grad=False).to(device)
        self.weighty = nn.Parameter(data=kernely, requires_grad=False).to(device)

    def forward(self, x):
        sobelx = F.conv2d(x, self.weightx, padding=1)
        sobely = F.conv2d(x, self.weighty, padding=1)
        return torch.abs(sobelx) + torch.abs(sobely)


class FILMFusionLoss(nn.Module):
    def __init__(
        self,
        prior: str = "mean",
        weight_intensity: float = 1.0,
        weight_grad: float = 300,  # [20, 300, 500] for VIF, MEF, MFF
        weight_ssim: float = 1.0,
        ssim_window_size: int = 5,
    ):
        super().__init__()
        self.weight_ssim = weight_ssim

        self.lp_ssim = LpLssimLossweight(window_size=ssim_window_size)
        self.fusion_loss = Fusionloss(
            coeff_int=weight_intensity, coeff_grad=weight_grad, in_max=prior == "max"
        )

    def color_convert(self, img):
        if img.shape[1] == 1:
            return img
        elif img.shape[1] == 3:
            return kornia.color.rgb_to_grayscale(img)
        else:
            raise ValueError(f"img should have 1 or 3 channels, but got {img.shape[1]}")

    def forward(self, img_fused, boundary_gt, fusion_gt=None, *args, **kwargs):
        # boundary_gt: cat([vi, ir])
        if isinstance(boundary_gt, (tuple, list)):
            img_vis, img_ir = boundary_gt
        elif torch.is_tensor(boundary_gt):
            assert boundary_gt.size(1) in [2, 4, 6]
            bc = boundary_gt.size(1)
            if bc == 2:
                img_vis, img_ir = boundary_gt[:, :1], boundary_gt[:, 1:]
            elif bc == 4:
                img_vis, img_ir = boundary_gt[:, :3], boundary_gt[:, 3:]
            else:
                img_vis, img_ir = boundary_gt[:, :3], boundary_gt[:, 3:]
        else:
            raise ValueError(
                f"boundary_gt should be a tensor or a tuple/list, but got {type(boundary_gt)}"
            )

        # convert to grayscale to compute loss
        img_vis = self.color_convert(img_vis)
        img_ir = self.color_convert(img_ir)
        img_fused = self.color_convert(img_fused)

        h, w = img_vis.shape[-2:]
        weight = math.floor((math.sqrt(h * w)))
        loss_d = {}
        loss_total = 0.0

        if self.weight_ssim > 0.0:
            # boundaries: lp, ssim
            lp_ssim_vis, lp_vis, lssim_vis = self.lp_ssim(img_vis, img_fused, weight)
            lp_ssim_ir, lp_ir, lssim_ir = self.lp_ssim(img_ir, img_fused, weight)
            loss_ssim = lp_ssim_vis + lp_ssim_ir
            loss_total += loss_ssim * self.weight_ssim

        # priors: total, prior, grad
        loss_in_grad, loss_in, loss_grad = self.fusion_loss(img_vis, img_ir, img_fused)
        loss_total += loss_in_grad

        # loss dict
        if self.weight_ssim > 0.0:
            loss_d["l_ssim"] = lssim_vis + lssim_ir
            loss_d["l_boundary"] = lp_vis + lp_ir
        # loss_d['l_prior'] = loss_in
        # loss_d['l_grad'] = loss_grad
        loss_d["l_in_grad"] = loss_in_grad
        loss_d["l_total"] = loss_total

        return loss_total, loss_d


# ====================================================== MaskGiT and Next Token Prediction Loss ======================================================
from typing import Mapping, Text, Tuple

from einops import rearrange


class MLMLoss(torch.nn.Module):
    def __init__(
        self, label_smoothing: float = 0.1, loss_weight_unmasked_token: float = 1.0
    ):
        super().__init__()
        self.label_smoothing = label_smoothing
        self.loss_weight_unmasked_token = loss_weight_unmasked_token
        self.criterion = torch.nn.CrossEntropyLoss(
            label_smoothing=self.label_smoothing, reduction="none"
        )

    def forward(
        self, inputs: torch.Tensor, targets: torch.Tensor, weights=None
    ) -> Tuple[torch.Tensor, Mapping[Text, torch.Tensor]]:
        inputs = rearrange(inputs, "b ... c -> b c ...")
        loss = self.criterion(inputs, targets)
        weights = weights.to(loss)
        loss_weights = (
            (1.0 - weights) * self.loss_weight_unmasked_token + weights
        )  # set 0 to self.loss_weight_unasked_token
        loss = (loss * loss_weights).sum() / (loss_weights.sum() + 1e-8)
        # we only compute correct tokens on masked tokens
        correct_tokens = ((torch.argmax(inputs, dim=1) == targets) * weights).sum(
            dim=1
        ) / (weights.sum(1) + 1e-8)
        return loss, {"loss": loss, "correct_tokens": correct_tokens.mean()}


class ARLoss(torch.nn.Module):
    def __init__(self, codebook_size: int):
        super().__init__()
        self.target_vocab_size = codebook_size
        self.criterion = torch.nn.CrossEntropyLoss(reduction="mean")

    def forward(
        self, logits: torch.Tensor, labels: torch.Tensor
    ) -> Tuple[torch.Tensor, Mapping[Text, torch.Tensor]]:
        shift_logits = logits[..., :-1, :].permute(0, 2, 1).contiguous()  # NLC->NCL
        shift_labels = labels.contiguous()
        shift_logits = shift_logits.view(
            shift_logits.shape[0], self.target_vocab_size, -1
        )
        shift_labels = shift_labels.view(shift_labels.shape[0], -1)
        shift_labels = shift_labels.to(shift_logits.device)
        loss = self.criterion(shift_logits, shift_labels)
        correct_tokens = (torch.argmax(shift_logits, dim=1) == shift_labels).sum(
            dim=1
        ) / shift_labels.size(1)
        return loss, {"loss": loss, "correct_tokens": correct_tokens.mean()}


# ======================================================= loss function getter =======================================================


def get_emma_fusion_loss(
    fusion_model: nn.Module, device: str = None, model_is_y_pred: bool = True
):
    from utils.utils_modules import TranslationUnet

    # color bg may cause object color shift, so we constraint on the whole image
    main_loss = DRMFFusionLoss(reduce_label=False, color_loss_bg_masked=True)
    if device is not None:
        main_loss = main_loss.to(device)
    else:
        main_loss = main_loss.cuda()

    return EMMAFusionLoss(
        fusion_model=fusion_model,
        to_source_A_model=TranslationUnet,
        to_source_B_model=TranslationUnet,
        A_pretrain_path="utils/ckpts/Av.pth",
        B_pretrain_path="utils/ckpts/Ai.pth",
        A_model_kwargs={},
        B_model_kwargs={},
        translation_kwargs={"shift_num": 0, "rotate_num": 1, "flip_num": 2},
        main_once_fusion_loss=main_loss,
        detach_fused=True,  # avoid GPU OOM
        translation_weight=6.0,
        refusion_weight=0.1,
        model_pred_y=model_is_y_pred,
    )


def get_loss(loss_type, channel=31, **kwargs):
    """
    get loss function by name

    """

    if loss_type == "mse":
        criterion = TorchLossWrapper((1.0,), mse=nn.MSELoss())
    elif loss_type == "l1":
        criterion = TorchLossWrapper((1.0,), l1=nn.L1Loss())
    elif loss_type == "hybrid":
        criterion = HybridL1L2()
    elif loss_type == "smoothl1":
        criterion = TorchLossWrapper((1.0,), smooth_l1=nn.SmoothL1Loss())
    elif loss_type == "l1ssim":
        criterion, _ = init_cls_with_kwargs(HybridL1SSIM, kwargs)
    elif loss_type == "ssimrmi_fuse":
        criterion = HybridSSIMRMIFuse(weight_ratio=(1, 1), ssim_channel=channel)
    elif loss_type == "pia_fuse":
        # perceptual loss should be less weighted
        criterion = HybridPIALoss(weight_ratio=(3, 7, 20, 10))
    elif loss_type == "charbssim":
        criterion = HybridCharbonnierSSIM(
            channel=kwargs.pop("channel", 1), weighted_r=(1.0, 1.0)
        )
    elif loss_type == "ssimsf":
        # YDTR loss
        # not hack weighted ratio
        criterion = HybridSSIMSF(channel=kwargs.pop("channel", 1))
    elif loss_type == "ssimmci":
        criterion = HybridSSIMMCI(channel=kwargs.pop("channel", 1))
    elif loss_type == "mcgmci":
        criterion = HybridMCGMCI(weight_r=(2.0, 1.0))
    elif loss_type == "u2fusion":
        criterion = U2FusionLoss()
    elif loss_type == "cddfusion":
        criterion = CDDFusionLoss(weights=(1.5, 1, 1))
    elif loss_type == "swinfusion":
        criterion = SwinFusionLoss()
    elif loss_type == "drmffusion":
        criterion, _ = init_cls_with_kwargs(DRMFFusionLoss, kwargs)
    elif loss_type == "emmafusion":
        criterion, _ = init_cls_with_kwargs(EMMAFusionLoss, kwargs)
    elif loss_type == "filmfusion":
        criterion, _ = init_cls_with_kwargs(FILMFusionLoss, kwargs)
    else:
        raise NotImplementedError(f"loss {loss_type} is not implemented")
    return criterion


def inspect_func_kwargs(func: callable, arg_dict: dict | None = None):
    log = easy_logger(func_name=f"init_{func.__name__}", level="DEBUG")

    if arg_dict is None:
        arg_dict = {}
    sig = inspect.signature(func)
    in_func_kwargs = {}
    _has_var_keywords = False

    for name, param in sig.parameters.items():
        if (
            (name not in arg_dict or name == "self")
            and param.kind
            in [  # to explicitly avoid func is class.__init__ and enforce `self` is provided by __new__ and do not need exists in arg_dict
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
            ]
        ):
            if param.default is not inspect.Parameter.empty:
                in_func_kwargs[name] = param.default
                log.debug(
                    f'keyword {name} is not provided, use default value "{param.default}"'
                )
            else:
                raise ValueError(
                    f"keyword {name} is not provided and no default value is set (raised by inspect._empty)"
                )
        elif param.kind == inspect.Parameter.VAR_KEYWORD:  # if func has **kwargs
            _has_var_keywords = True
        else:
            in_func_kwargs[name] = arg_dict.pop(name)

    if not _has_var_keywords:
        if len(arg_dict) > 0:
            logger.warning(
                f"init function {func.__name__} got unused key words {arg_dict}"
            )
        return in_func_kwargs, arg_dict
    else:
        in_func_kwargs.update(arg_dict)
        return in_func_kwargs, dict()


def init_cls_with_kwargs(cls: type, arg_dict: dict):
    # class with __init__ or a callable function
    if hasattr(cls, "__init__") or inspect.isfunction(cls) or inspect.isclass(cls):
        kwargs, remained_kwargs = inspect_func_kwargs(cls, arg_dict)
    elif hasattr(cls, "__new__"):
        # support singleton pattern
        raise NotImplementedError("singleton pattern is not supported yet")
    else:
        raise ValueError(
            f"cls {cls} should have __init__ or __new__ method with kwargs {arg_dict}"
        )

    return cls(**kwargs), remained_kwargs


if __name__ == "__main__":
    # loss = SSIMLoss(channel=31)
    # loss = CharbonnierLoss(eps=1e-3)
    # x = torch.randn(1, 31, 64, 64, requires_grad=True)
    # y = x + torch.randn(1, 31, 64, 64) / 10
    # l = loss(x, y)
    # l.backward()
    # print(l)
    # print(x.grad)

    # import PIL.Image as Image

    # vi = (
    #     np.array(
    #         Image.open(
    #             "/media/office-401/Elements SE/cao/ZiHanCao/datasets/RoadScene_and_TNO/training_data/vi/FLIR_05857.jpg"
    #         ).convert("L")
    #     )
    #     / 255
    # )
    # ir = (
    #     np.array(
    #         Image.open(
    #             "/media/office-401/Elements SE/cao/ZiHanCao/datasets/RoadScene_and_TNO/training_data/ir/FLIR_05857.jpg"
    #         ).convert("L")
    #     )
    #     / 255
    # )

    # torch.cuda.set_device("cuda:0")

    # vi = torch.tensor(vi)[None, None].float()  # .cuda()
    # ir = torch.tensor(ir)[None, None].float()  # .cuda()

    # fuse = ((vi + ir) / 2).repeat_interleave(2, dim=0)
    # fuse.requires_grad_()
    # print(fuse.requires_grad)

    # gt = torch.cat((vi, ir), dim=1).repeat_interleave(2, dim=0)

    # fuse_loss = HybridSSIMRMIFuse(weight_ratio=(1.0, 1.0, 1.0), ssim_channel=1)

    # from kornia.io import load_image, ImageLoadType

    # torch.cuda.set_device("cuda:1")

    # class FuseModel:
    #     def only_fusion_step(self, a, b):
    #         return a + b

    # print(get_loss('l1ssim', implem_by='kornia', window_size=11, grad_norm=False))

    # fuse_loss = DRMFFusionLoss(grad_op='sobel', reduce_label=True, pseudo_l1_const=0., grad_norm=True).cuda()
    # fuse_loss = get_emma_fusion_loss(FuseModel())
    # print(fuse_loss(fused, (vis, ir), mask))

    # u2fusion_loss = U2FusionLoss().cuda()

    torch.cuda.set_device("cuda:1")
    from kornia.io import ImageLoadType, load_image

    loss = get_loss(
        "filmfusion",
        pred_mode="max",
        weight_intensity=1,
        weight_grad=300,
        weight_ssim=1.0,
        ssim_window_size=5,
    )

    vis = load_image(
        "/Data3/cao/ZiHanCao/datasets/MEF-SICE/under/178.jpg",
        ImageLoadType.UNCHANGED,
        device="cuda",
    )
    ir = load_image(
        "/Data3/cao/ZiHanCao/datasets/MEF-SICE/over/178.jpg",
        ImageLoadType.UNCHANGED,
        device="cuda",
    )
    fused = load_image(
        "/Data3/cao/ZiHanCao/exps/IF-FILM-main/test_output/SICE/recolored/178.jpg",
        ImageLoadType.UNCHANGED,
        device="cuda",
    )

    vis, ir, fused = (
        vis[None].float() / 255.0,
        ir[None].float() / 255.0,
        fused[None].float() / 255.0,
    )

    loss = loss(fused, (ir, vis))
    print(loss)

    # import time
    # fused = torch.randn(1, 3, 64, 64).cuda().requires_grad_()
    # vis = torch.randn(1, 3, 64, 64).cuda().requires_grad_()
    # ir = torch.randn(1, 3, 64, 64).cuda().requires_grad_()
    # mask = torch.randint(0, 3, (1, 1, 64, 64)).cuda().float()

    # to_tensor = lambda x: x[None].float() / 255.
    # fused = load_image("/Data3/cao/ZiHanCao/exps/panformer/visualized_img/RWKVFusion_v11_RWKVFusion/sice_v2/011.jpg", ImageLoadType.UNCHANGED, device='cuda:1')
    # vis = load_image("/Data3/cao/ZiHanCao/datasets/MEF-SICE/over/011.jpg", ImageLoadType.UNCHANGED, device='cuda:1')
    # ir = load_image("/Data3/cao/ZiHanCao/datasets/MEF-SICE/under/011.jpg", ImageLoadType.UNCHANGED, device='cuda:1')

    # fused, vis, ir = to_tensor(fused), to_tensor(vis), to_tensor(ir)
    # fused.requires_grad_()

    # loss = fuse_loss(fused, (vis, ir), mask=mask)
    # loss[0].backward()
    # print(loss)
    # time.sleep(0.1)

    # print(fused.grad)

    # fuse_loss = HybridPIALoss().cuda(1)
    # fuse_loss = CDDFusionLoss()  # .cuda()
    # loss, loss_d = fuse_loss(fuse, gt)
    # loss.backward()
    # print(loss)
    # print(loss_d)

    # print(fuse.grad)

    # mcg_mci_loss = HybridMCGMCI()
    # print(mcg_mci_loss(fuse, gt))
