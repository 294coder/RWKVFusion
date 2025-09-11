# GPL License
# Copyright (C) UESTC
# All Rights Reserved
#
# @Time    : 2023/4/10 12:37
# @Author  : Zihan Cao
# @reference:
#   metrics are modified by Zihan Cao
#

##============================================= history ======================================================##
#    Date         Author        Description
# 2024.10.1     Zihan Cao     add VIF, MEF, MFF metrics analysis and rgb and ycbcr color space metrics analysis.
# 2024.10.26    Zihan Cao     add CLIPIQA and MEFSSIM metric analysis.
# 2025.02.17    Zihan Cao     add unreference metrics: BRISQUE, MUSIQ, NIQE metric analysis.


##============================================================================================================##
import copy
import enum
import sys
import warnings
from contextlib import contextmanager
from functools import lru_cache, partial
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Sequence, Tuple, Union

import kornia
import numpy as np
import torch
from kornia.color.ycbcr import rgb_to_y
from natsort import natsorted
from rich.progress import track
from torch import Tensor
from torch.func import vmap
from torchvision.io import ImageReadMode, read_image
from tqdm import tqdm
from typing_extensions import TypeAlias

from utils.log_utils import catch_any_error


def rgb_to_ycbcr(image: torch.Tensor) -> torch.Tensor:
    """
    Convert an RGB image to YCbCr.

    Args:
        image: RGB image tensor with shape (..., 3, H, W) in range [0, 1]

    Returns:
        YCbCr image tensor with shape (..., 3, H, W)
    """
    if not isinstance(image, torch.Tensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(image)}")

    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError(
            f"Input size must have a shape of (..., 3, H, W). Got {image.shape}"
        )

    r: torch.Tensor = image[..., 0, :, :]
    g: torch.Tensor = image[..., 1, :, :]
    b: torch.Tensor = image[..., 2, :, :]

    y: torch.Tensor = 0.29900 * r + 0.58700 * g + 0.11400 * b
    cb: torch.Tensor = -0.168736 * r - 0.331264 * g + 0.50000 * b + 0.5
    cr: torch.Tensor = 0.50000 * r - 0.418688 * g - 0.081312 * b + 0.5

    return torch.stack([y, cb, cr], dim=-3)


def ycbcr_to_rgb(image: torch.Tensor) -> torch.Tensor:
    """
    Convert a YCbCr image to RGB.

    Args:
        image: YCbCr image tensor with shape (..., 3, H, W)

    Returns:
        RGB image tensor with shape (..., 3, H, W) in range [0, 1]
    """
    if not isinstance(image, torch.Tensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(image)}")

    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError(
            f"Input size must have a shape of (..., 3, H, W). Got {image.shape}"
        )

    y: torch.Tensor = image[..., 0, :, :]
    cb: torch.Tensor = image[..., 1, :, :]
    cr: torch.Tensor = image[..., 2, :, :]

    r: torch.Tensor = y + 1.40200 * (cr - 0.5)
    g: torch.Tensor = y - 0.34414 * (cb - 0.5) - 0.71414 * (cr - 0.5)
    b: torch.Tensor = y + 1.77200 * (cb - 0.5)

    return torch.stack([r, g, b], dim=-3).clamp(0, 1)


##############################################################################################################
######################################## VIF, MEF, MFF metric analysis #######################################
##############################################################################################################
from typing import Literal

from utils.fusion_index import (
    evaluate_MEF_MFF_metric_torch,
    evaluate_VIF_metric_numpy,
    evaluate_VIF_metric_numpy_old,
    evaluate_VIF_metric_torch,
)
from utils.log_utils import easy_logger
from utils.misc import default, dict_to_str

if TYPE_CHECKING:
    if sys.version_info <= (3, 11):
        predType: TypeAlias = Union[Tensor, np.ndarray]
        gtType: TypeAlias = Union[
            Tensor,
            Tuple[Tensor],
            Dict[str, Tensor],
            np.ndarray,
            Sequence[Tensor, Tensor],
            Sequence[np.ndarray, np.ndarray],
            Dict[str, np.ndarray],
        ]
    elif sys.version_info >= (3, 11):
        predType: TypeAlias = Tensor | np.ndarray
        gtType: TypeAlias = (
            Tensor
            | tuple[Tensor]
            | dict[str, Tensor]
            | np.ndarray
            | Sequence[Tensor, Tensor]
            | Sequence[np.ndarray, np.ndarray]
            | dict[str, np.ndarray]
        )


logger = easy_logger(func_name="AnalysisFusionAcc")

_NEW_METRICS_VIF = [
    "EN",
    "SD",
    "MI",
    "VIF",
    "SF",
    "AG",
    "MSE",
    "CC",
    "PSNR",
    "SCD",
    "Qabf",
    "SSIM",
]
_NEW_METRICS_MEF_MFF = [
    "MEFSSIM",
    "Qc",
    "Qp",
    "Qcb",
    "Qcv",
    "Qw",
    "Qy",
    "Qncie",
    "NMI",
    "LPIPS",
    "FID",
    "CLIPIQA",
    "BRISQUE",
    "MUSIQ",
    "NIQE",
]
_NEW_METRICS_ALL = _NEW_METRICS_VIF + _NEW_METRICS_MEF_MFF
_ON_TRAIN_METRICS = [
    "EN",
    "SD",
    "MI",
    "VIF",
    "SF",
    "AG",
    "MSE",
    "CC",
    "PSNR",
    "SCD",
    "Qabf",
    "SSIM",
    "Qc",
    "Qp",
    "Qcb",
    "Qcv",
    "Qw",
    "Qy",
    "Qncie",
    "NMI",
]
_OLD_METRICS = ["PSNR", "EN", "SD", "SF", "AG", "SSIM", "VIF"]

_METRICS_BETTER_MAPPING = {
    "EN": "↑",
    "SD": "↑",
    "SF": "↑",
    "AG": "↑",
    "MI": "↑",
    "MSE": "↓",
    "CC": "↑",
    "PSNR": "↑",
    "SCD": "↑",
    "VIF": "↑",
    "Qabf": "↑",
    "SSIM": "↑",
    "Qc": "↑",
    "Qp": "↑",
    "Qcb": "↑",
    "Qcv": "↓",
    "Qw": "↑",
    "Qncie": "↑",
    "Qy": "↑",
    "NMI": "↑",
    "FMI": "↑",
    "LPIPS": "↓",
    "FID": "↓",
    "MEFSSIM": "↑",
    "CLIPIQA": "↑",
    "BRISQUE": "↓",
    "MUSIQ": "↑",
    "NIQE": "↓",
}


class MetricsByTask(enum.Enum):
    VIF = "vif"
    MEF_MFF = "mef_mff"
    ALL = "all"
    ON_TRAIN = "on_train"
    OLD_METRICS = "old"
    CUSTOM = "custom"


class AnalysisFusionAcc(object):
    def __init__(
        self,
        unorm: bool = True,
        legacy_metric: bool = False,
        progress_bar: bool = False,
        test_metrics: str | list[str] = MetricsByTask.ALL,
        implem_by: Literal["torch", "numpy"] = "torch",
        only_on_y_component: bool = True,
        *,
        results_decimals: int = 4,
        rgb_or_ycbcr_metric: Literal["rgb", "ycbcr"] = "rgb",  # remove the argument
    ):
        self.unorm_factor = 255 if unorm else 1  # must be 255
        if self.unorm_factor != 255:
            logger.warning(
                "image range should be [0, 255] for VIF metric, "
                + "but got unorm_factor={self.unorm_factor}."
            )
        self.legacy_metric = legacy_metric
        self._tested_metrics = test_metrics
        self.implem_by = implem_by
        self.only_on_y_component = only_on_y_component
        self.rgb_or_ycbcr_metric = rgb_or_ycbcr_metric
        if only_on_y_component:
            logger.info("metrics are computed on Y component")
        elif rgb_or_ycbcr_metric == "rgb":
            logger.info("metrics are computed on RGB channels")
        elif rgb_or_ycbcr_metric == "ycbcr":
            assert False, "unsupported yet"
            logger.info("metrics are computed on YCbCr channels")
        else:
            raise ValueError(
                f'`rgb_or_ycbcr_metric` should be "rgb" or "ycbcr", but got {rgb_or_ycbcr_metric}'
            )

        if self.legacy_metric:
            logger.warning(
                "using legacy metric which is not recommended and it is implemented by numpy which is slow"
            )
            self.metric_VIF_fn = evaluate_VIF_metric_numpy_old
        else:
            if implem_by == "torch":
                self._metric_VIF_fn = evaluate_VIF_metric_torch
                self._metric_MEF_MFF_fn = evaluate_MEF_MFF_metric_torch
                self.metric_VIF_fn = partial(
                    self._metric_VIF_fn, metrics=self.tested_metrics
                )
                self.metric_MEF_MFF_fn = partial(
                    self._metric_MEF_MFF_fn, metrics=self.tested_metrics
                )
            else:
                logger.warning(
                    f"using numpy implementation for metric which is [i]slow[/i]."
                )
                logger.warning(
                    "numpy implementaion may differ from torch implementaion,",
                    "we recommend using torch implementaion.",
                )
                self._metric_VIF_fn = partial(
                    evaluate_VIF_metric_numpy, metrics=self.tested_metrics
                )
                self.metric_VIF_fn = self._metric_VIF_fn
        self.progress_bar = progress_bar
        self.results_decimals = results_decimals

        # acc tracker
        self._sumed_acc = {}
        self._acc_ave = {}
        self._acc_d = {}
        self._call_n = 0

        # reinit metric flag
        self._reinit_metric_flag = False

    def clear(self):
        logger.info("clear metric tracker")

        # acc tracker
        self._sumed_acc = {}
        self._acc_ave = {}
        self._acc_d = {}
        self._call_n = 0

        # re-init metric flag
        self._reinit_metric_flag = False

    @property
    def tested_metrics(self):
        metrics_mapping = {
            MetricsByTask.VIF: _OLD_METRICS if self.legacy_metric else _NEW_METRICS_VIF,
            MetricsByTask.MEF_MFF: _NEW_METRICS_MEF_MFF,
            MetricsByTask.ALL: _NEW_METRICS_ALL,
            MetricsByTask.OLD_METRICS: _OLD_METRICS,
            MetricsByTask.ON_TRAIN: _ON_TRAIN_METRICS,
            MetricsByTask.CUSTOM: self._tested_metrics,
        }

        # copy it from remove or any change
        if isinstance(self._tested_metrics, (str, MetricsByTask)):
            return copy.deepcopy(metrics_mapping[self._tested_metrics])
        else:
            return copy.deepcopy(self._tested_metrics)

    @property
    def reinit_metric_flag(self):
        if not self._reinit_metric_flag:
            self._reinit_metric_flag = True
            return True
        return False

    @property
    def acc_ave(self):
        acc_ave = {}
        for k, v in self._acc_ave.items():
            if not callable(v):
                acc_ave[k] = v.item()
            else:
                # keep just one function to call
                acc_ave[k] = v().item()
        return acc_ave

    @property
    def sumed_acc(self):
        acc_sum = {}
        for k, v in self._sumed_acc.items():
            if not callable(v):
                acc_sum[k] = v.item()
            else:
                # keep just one function to call
                acc_sum[k] = v().item() * self._call_n

        return acc_sum

    @property
    def metrics_better_order(self):
        return [_METRICS_BETTER_MAPPING[m] for m in self.tested_metrics]

    def _average_acc(self, d_ave, n):
        for k in d_ave.keys():
            d_ave[k] /= n
        return d_ave

    @property
    def empty_acc(self):
        return {k: 0.0 for k in self.tested_metrics}

    def drop_dim(self, x: "torch.Tensor", to_numpy=False):
        """
        [1, h, w] -> [h, w]
        [c, h, w]: unchange
        """
        assert x.ndim in (
            2,
            3,
        ), f"must be 2 or 3 number of dimensions, but got {x.ndim}"

        if x.size(0) == 1:
            x = x.squeeze(0)

        if to_numpy and torch.is_tensor(x):
            return x.detach().cpu().numpy()
        else:
            return x

    @staticmethod
    def dict_items_sum(b_acc_d: list):
        sum_d = {}
        for acc_d in b_acc_d:
            for k, v in acc_d.items():
                if not callable(v):
                    sum_d[k] = sum_d.get(k, 0) + v
                else:
                    # keep just one function to call
                    sum_d[k] = v
        return sum_d

    @staticmethod
    def average_all(sum_d: dict, bs: int, prev_call_n: int, prev_sumed_acc: dict):
        call_n = prev_call_n + bs
        summed_acc = {}
        acc_ave = {}

        # for each metric
        for k, v in sum_d.items():
            if not callable(v):
                summed_acc[k] = prev_sumed_acc.get(k, 0) + v
                acc_ave[k] = summed_acc[k] / call_n
            else:
                # keep just one function to call
                summed_acc[k] = v
                acc_ave[k] = v

        return summed_acc, acc_ave, call_n

    def _color_to_Y(self, x: "Tensor"):
        # x: batched image, shaped as [b, 3, h, w] or [b, 1, h, w]

        if x.size(1) == 3:
            return rgb_to_y(x)
        elif x.size(1) == 1:
            return x
        else:
            raise ValueError(f"x.size(1) should be 1 or 3, but got {x.size(1)}")

    def _gray_to_color(self, x: "Tensor"):
        # x: batched image, shaped as [b, 1, h, w]
        if x.ndim == 4 and x.size(1) == 1:
            return x.repeat(1, 3, 1, 1)
        elif x.ndim == 3 and x.size(0) == 1:
            return x.unsqueeze(0).repeat(3, 1, 1)
        elif x.ndim == 2:
            return x.unsqueeze(0).unsqueeze(0).repeat(3, 1, 1)
        elif x.ndim == 4 and x.size(1) == 3:
            return x
        else:
            raise ValueError(
                f"x should be shaped as [b, 1, h, w] or [b, 3, h, w] or [1, h, w], "
                + f"but got {x.shape}"
            )

    def _color_to_YCbCr(self, x: "Tensor"):
        # x: batched image, shaped as [b, 3, h, w]
        return rgb_to_ycbcr(x)

    def type_color_check(self, x: "Tensor | tuple[Tensor]", check_color: bool = True):
        # x: batched image, shaped as [b, c, h, w]

        def _inner_type_check(x):
            if torch.is_tensor(x):
                if x.dtype == torch.uint8:
                    assert (
                        self.unorm_factor == 1
                    ), f"unorm_factor should be 1 for uint8 input, but got {self.unorm_factor}"
                    x = x.float()

                # color conversion
                if check_color:
                    if self.only_on_y_component and x.size(1) == 3:
                        # ir -> ir; vi -> vi_y
                        # over -> over_y; under -> under_y
                        x = self._color_to_Y(x)
                    elif not self.only_on_y_component and x.size(1) == 1:
                        # ir -> ir_rgb; vi -> vi_rgb
                        # over -> over; under -> under
                        x = self._gray_to_color(x)

                    # TODO: need test
                    elif self.rgb_or_ycbcr_metric == "ycbcr":
                        assert False, "unsupported yet"
                        # dtype should be uint8
                        x = self._color_to_YCbCr(x)
                    elif self.rgb_or_ycbcr_metric == "rgb":
                        # do nothing
                        pass
                    else:
                        raise ValueError(
                            f"rgb_or_ycbcr_metric should be rgb or ycbcr, but got {self.rgb_or_ycbcr_metric}"
                        )

            return x

        if isinstance(x, (tuple, list)):
            typed_x = []
            for xi in x:
                typed_x.append(_inner_type_check(xi))
            return typed_x
        else:
            return _inner_type_check(x)

    @staticmethod
    def _get_type_check(gt: "torch.Tensor | tuple | list | dict", pred: torch.Tensor):
        fusion_chans = pred.size(1)

        if isinstance(gt, Tensor):
            gt_chans = gt.size(1)
            assert gt.shape[-2:] == pred.shape[-2:], (
                f"gt and pred should have same shape,"
                f"but got gt.shape[-2:]=={gt.shape[-2:]}, pred.shape[-2:]=={pred.shape[-2:]}"
            )
            assert (
                gt_chans == 2 or gt_chans == 4 or gt_chans == 6
            ), f"gt.size(1) should be 2, 4 or 6, but got {gt.size(1)}"
            vi, ir = gt.split([fusion_chans, gt_chans - fusion_chans], dim=1)
        elif isinstance(gt, (tuple, list)):
            assert len(gt) == 2, f"gt should have 2 element, but got {len(gt)}"
            assert gt[0].shape[-2:] == pred.shape[-2:], (
                f"gt[0] and pred should have same shape,"
                f"but got gt[0].shape[-2:]=={gt[0].shape[-2:]}, pred.shape[-2:]=={pred.shape[-2:]}"
            )
            assert gt[1].shape[-2:] == pred.shape[-2:], (
                f"gt[1] and pred should have same shape,"
                f"but got gt[1].shape[-2:]=={gt[1].shape[-2:]}, pred.shape[-2:]=={pred.shape[-2:]}"
            )
            vi, ir = gt
        elif isinstance(gt, dict):
            vi = gt["vi"]
            ir = gt["ir"]
            assert vi.shape[-2:] == pred.shape[-2:], (
                f"vi and pred should have same shape,"
                f"but got vi.shape[-2:]=={vi.shape[-2:]}, pred.shape[-2:]=={pred.shape[-2:]}"
            )
            assert ir.shape[-2:] == pred.shape[-2:], (
                f"ir and pred should have same shape,"
                f"but got ir.shape[-2:]=={ir.shape[-2:]}, pred.shape[-2:]=={pred.shape[-2:]}"
            )
        else:
            raise ValueError(
                f"gt should be Tensor or tuple of Tensor, but got {type(gt)}"
            )

        return vi, ir

    def _one_batch_call_MEF_MFF(
        self,
        gt: "gtType",
        pred: "predType",
    ):
        # * MEF and MFF metrics are computed on RGB channels
        # * VIF with only 1-channel NIR image will only compute metrics on Y component (a.k.a gray image)

        b = pred.size(0)
        fusion_chans = pred.size(1)
        unorm_fn = lambda x: (x * self.unorm_factor).clip(0, 255).type(torch.float32)

        vi, ir = self._get_type_check(gt, pred)
        is_three_chan = fusion_chans == 3 and not self.only_on_y_component
        if is_three_chan:
            vi, ir, pred = map(partial(self.type_color_check), (vi, ir, pred))

        vi, ir, pred = map(unorm_fn, (vi, ir, pred))

        batch_acc_d = []
        tbar = tqdm(
            zip(vi, ir, pred),
            total=b,
            disable=not self.progress_bar,
            leave=False,
            desc="MEF_MFF task evaluation ...",
        )

        # if test metrics contains LPIPS and FID, we compute them on 3 channels
        _fallback_metrics = [
            "Qncie",
            "NMI",
            "MEFSSIM",
        ]  # does not support vmap implementation
        _default_non_vmap_metrics = [
            "LPIPS",
            "FID",
            "CLIPIQA",
            "BRISQUE",
            "MUSIQ",
            "NIQE",
        ]
        rgb_non_vmap_metrics = []
        trad_metrics = self.tested_metrics  # constructed from self.tested_metrics

        # remove perceptual metrics from traditional metrics
        for pm in _default_non_vmap_metrics:
            if pm in self.tested_metrics:
                rgb_non_vmap_metrics.append(pm)
                trad_metrics.remove(pm)

        # vmap fallback metrics
        for m in _fallback_metrics:
            if m in trad_metrics:
                trad_metrics.remove(m)

        # import ipdb; ipdb.set_trace()
        # * ==========================================================
        # * for loop on batch dimension

        for vi_i, ir_i, f_i in tbar:
            acc_d = {}
            if is_three_chan:
                # * ============== for traditional metrics ==============
                # we compute traditional metrics on each channel
                ##* vmap fast implementation
                metric_trad_fns = partial(self.metric_MEF_MFF_fn, metrics=trad_metrics)
                dict_MEF_MFF_trad = vmap(metric_trad_fns)(vi_i, ir_i, f_i)
                acc_d.update(
                    {m: value.mean() for m, value in dict_MEF_MFF_trad.items()}
                )

                ##* pythonic for-loop
                _any_to_fallback = any(
                    [m in _fallback_metrics for m in self.tested_metrics]
                )
                _acc_ds = []
                if _any_to_fallback:
                    for j in range(3):
                        # traditional metrics
                        dict_MEF_MFF_trad = self.metric_MEF_MFF_fn(
                            vi_i[j], ir_i[j], f_i[j], metrics=_fallback_metrics
                        )
                        _acc_ds.append(dict_MEF_MFF_trad)
                    acc_d2 = self._mean_dict(_acc_ds)
                    acc_d.update(acc_d2)
            else:
                # * ========================== for one-channel traditional metrics ==========================
                # * since one channel image, we only compute traditional metrics
                # * and ignore perceptual metrics
                # * =========================================================================================

                trad_metrics = self.tested_metrics

                # has any perceptual metrics
                _any_percep_metrics = any(
                    [pm in trad_metrics for pm in _default_non_vmap_metrics]
                )
                if _any_percep_metrics:
                    # warnings.warn('MEF_MFF metrics only support 3 channels, so only compute metrics on Y channel, ' + \
                    #               'and ignore LPIPS, FID, and CLIPIQA metrics')
                    trad_metrics = [
                        m for m in trad_metrics if m not in _default_non_vmap_metrics
                    ]

                acc_d = self.metric_MEF_MFF_fn(
                    vi_i[0], ir_i[0], f_i[0], metrics=trad_metrics
                )

            # * ============== for perceptual metrics ==============
            # we compute perceptual metrics on 3 channels
            if len(rgb_non_vmap_metrics) > 0 and fusion_chans == 3:
                dict_MEF_MFF_percep = self.metric_MEF_MFF_fn(
                    vi_i,
                    ir_i,
                    f_i,
                    metrics=rgb_non_vmap_metrics,
                    reinit_cls_metrics=self.reinit_metric_flag,
                )
                acc_d.update(dict_MEF_MFF_percep)

            # add in batched metric dict
            batch_acc_d.append(acc_d)

        sum_d = self.dict_items_sum(batch_acc_d)

        return sum_d

    def _one_batch_call_VIF(self, gt: "gtType", pred: "predType"):
        """call the metric function for one batch

        Args:
            gt (Tensor | tuple[Tensor]): Tensor by catting the vis and ir, or tuple of vis and ir;
            channel for `Tensor` type should be 1+1 or 3+1 (rgb and infared). If tuple, assumed to be (vis, ir).
            pred (Tensor): fused image shaped as [b, 1, h, w] or [b, 3, h, w].
        """
        b = pred.size(0)
        fusion_chans = pred.size(1)
        if self.only_on_y_component:
            is_three_chan = False
        else:
            is_three_chan = fusion_chans == 3
        unorm_fn = lambda x: (x * self.unorm_factor).clip(0, 255).type(torch.float32)

        # * may convert RGB to Y
        vi, ir = self._get_type_check(gt, pred)
        vi, ir, pred = map(self.type_color_check, (vi, ir, pred))
        vi, ir, pred = map(unorm_fn, (vi, ir, pred))

        batch_acc_d = []
        tbar = tqdm(
            zip(vi, ir, pred),
            total=b,
            disable=not self.progress_bar,
            leave=False,
            desc="VIF task evaluation ...",
        )
        for vi_i, ir_i, f_i in tbar:
            if self.legacy_metric:
                # TODO: need test
                assert not is_three_chan, "legacy metric only support gray input"
                vi_i, ir_i, f_i = map(self.drop_dim, (vi_i, ir_i, f_i))
                acc_d = self.metric_VIF_fn(vi_i, ir_i, f_i)

            # * ===========================================================================
            # * we recommend using torch implementaion for a fast metric evaluation
            # TODO: MEF and MFF metrics must be computed on RGB channels

            # adpoted from Zixiang Zhao and MEFB github repository

            else:
                _to_numpy = True if self.implem_by == "numpy" else False

                ## compute metrics on rgb channels
                if is_three_chan:
                    # logger.info('VIF: is three channels')
                    _metrics_vmap = [
                        m for m in self.tested_metrics if m not in ["MI", "VIF"]
                    ]
                    metric_fn = partial(self.metric_VIF_fn, metrics=_metrics_vmap)

                    ##* vmap implementation (do not support MI and VIF metrics, since they are in dynamic shape)
                    def _vmap_loop(fi, vi_i, ir_i):
                        fused = self.drop_dim(fi, to_numpy=_to_numpy)
                        vis = self.drop_dim(vi_i, to_numpy=_to_numpy)
                        ir = self.drop_dim(ir_i, to_numpy=_to_numpy)
                        return metric_fn(vis, ir, fused)

                    # pack in dim 0 (not allowed to call .item() on the result)
                    dict_VIF = vmap(_vmap_loop)(f_i, vi_i, ir_i)

                    # mean the metrics
                    acc_d = {m: value.mean() for m, value in dict_VIF.items()}

                    ##* pythonic loop for MI and VIF metrics (fall back to pythonic loop)
                    if "MI" in self.tested_metrics or "VIF" in self.tested_metrics:
                        _acc_ds = []
                        _metrics_py_loop = [
                            m for m in ["MI", "VIF"] if m in self.tested_metrics
                        ]
                        for j in range(3):
                            fused, vis, ir = map(
                                partial(self.drop_dim, to_numpy=_to_numpy),
                                (f_i[j], vi_i[j], ir_i[j]),
                            )
                            dict_VIF = self._metric_VIF_fn(
                                vis, ir, fused, metrics=_metrics_py_loop
                            )
                            _acc_ds.append(dict_VIF)

                        acc_d2 = self._mean_dict(_acc_ds)
                        acc_d.update(acc_d2)

                ## compute metrics on Y channel
                else:
                    # logger.info('VIF: is one channel')
                    fused, vis, ir = map(
                        partial(self.drop_dim, to_numpy=_to_numpy),
                        (f_i[0], vi_i[0], ir_i[0]),
                    )
                    dict_VIF = self.metric_VIF_fn(vis, ir, fused)
                    acc_d = dict_VIF

            # * ===========================================================================
            batch_acc_d.append(acc_d)

        sum_d = self.dict_items_sum(batch_acc_d)

        return sum_d

    def one_batch_call(self, gt: "gtType", pred: "predType"):
        # batch tensors --(sum_up)-> self._acc_d --(mean)-> self.acc_ave

        b = pred.size(0)

        if self.legacy_metric:
            VIF_dict = self._one_batch_call_VIF(gt, pred)
            self._acc_d = VIF_dict
        else:
            _is_custom_metrics = isinstance(self._tested_metrics, list)
            if self._tested_metrics == MetricsByTask.VIF:
                VIF_dict = self._one_batch_call_VIF(gt, pred)
                self._acc_d = VIF_dict
            elif self._tested_metrics == MetricsByTask.MEF_MFF:
                MEF_MFF_dict = self._one_batch_call_MEF_MFF(gt, pred)
                self._acc_d = MEF_MFF_dict
            elif (
                self._tested_metrics in [MetricsByTask.ALL, MetricsByTask.ON_TRAIN]
                or _is_custom_metrics
            ):
                VIF_dict = self._one_batch_call_VIF(gt, pred)
                MEF_MFF_dict = self._one_batch_call_MEF_MFF(gt, pred)
                self._acc_d = VIF_dict | MEF_MFF_dict
            else:
                raise ValueError(
                    f"self._tested_metrics should be VIF, MEF_MFF or ALL, but got {self._tested_metrics}"
                )

        self._sumed_acc, self._acc_ave, self._call_n = self.average_all(
            self._acc_d, b, self._call_n, self._sumed_acc
        )

    @staticmethod
    def _mean_dict(d: "list[dict]"):
        mean_d = {}
        keys = d[0].keys()
        for k in keys:
            for d_i in d:
                value = d_i[k]
                if k not in mean_d:
                    mean_d[k] = []

                mean_d[k].append(value)
            mean_d[k] = sum(mean_d[k]) / len(d)

        return mean_d

    def __call__(self, gt: "gtType", pred: "predType"):
        self.one_batch_call(gt, pred)

    def result_str(
        self,
        acc_ave: dict | None = None,
        to_table: bool = False,
        *,
        table_type: str = "vertical",
    ):
        acc_ave_to_str = default(
            acc_ave, self.acc_ave
        )  # if acc_ave is_provided, use it, otherwise use self.acc_ave
        if not to_table:
            return dict_to_str(self.acc_ave, decimals=self.results_decimals)
        else:
            import io

            from rich.console import Console
            from rich.table import Table

            if table_type == "vertical":
                table = Table(title="AnalysisVISIRAcc")
                table.add_column("Metric")
                table.add_column("Value")
                for k, v in acc_ave_to_str.items():
                    table.add_row(k, str(round(v, 3)))
            elif table_type == "horizontal":
                table = Table(title="AnalysisVISIRAcc")
                table.add_column("Metrics")
                metrics = list(acc_ave_to_str.keys())
                values = list(acc_ave_to_str.values())
                for metric in metrics:
                    table.add_column(metric)

                row_values = ["Value"] + [str(round(value, 3)) for value in values]
                table.add_row(*row_values)
            else:
                raise ValueError(
                    f"table_type should be vertical or horizontal, but got {table_type}"
                )

            str_buf = io.StringIO()
            Console(file=str_buf, record=True).print(table)

            return str_buf.getvalue()

    def __repr__(self):
        return (
            f"AnalysisFusionAcc(unorm_factor={self.unorm_factor}, legacy_metric={self.legacy_metric}) \n"
            + f"Metrics: {self.tested_metrics} \n"
            + f"Current result: {self.result_str(to_table=True)}"
        )  # Updated to use to_table=True

    @property
    def last_acc(self):
        return self._acc_d

    def ave_result_with_other_analysors(
        self,
        analysors: "Union[list[AnalysisFusionAcc], AnalysisFusionAcc]",
        ave_to_self: bool = False,
    ):
        if isinstance(analysors, AnalysisFusionAcc):
            analysors = [analysors]

        for analysor in analysors:
            assert isinstance(
                analysor, AnalysisFusionAcc
            ), f"analysors should be list of `AnalysisVISIRAcc`, but got {type(analysor)}"
            assert analysor._call_n > 0, "analysor should be called at least once"
            assert (
                list(analysor.acc_ave.keys()) == list(self.acc_ave.keys())
            ), f"analysor should have same keys, but one is {list(analysor.acc_ave.keys())}, \
                                                                                 the other is {list(self.acc_ave.keys())}"

        sum_d = self.dict_items_sum([a.sumed_acc for a in analysors])
        call_n_times = sum([a._call_n for a in analysors])
        sumed_acc, acc_ave, call_n = self.average_all(
            sum_d, call_n_times, self._call_n, self.sumed_acc
        )

        if ave_to_self:
            self.sumed_acc = sumed_acc
            self.acc_ave = acc_ave
            self._call_n = call_n

        return acc_ave


@catch_any_error
@torch.no_grad()
def analysis_by_dir(
    m1_dir: str,
    m2_dir: str,
    fused_dir: str,
    verbose: bool = False,
    analyser: AnalysisFusionAcc = None,  # FIXME: clear analyser history may cause the metrics missing in the next run
    *,
    fusion_analyser_kwargs: dict = {},
):
    default_kwargs = dict(
        unorm=True,
        test_metrics=MetricsByTask.ALL,
        only_on_y_component=False,
    )

    if analyser is None:
        if len(fusion_analyser_kwargs) == 0:
            analyser = AnalysisFusionAcc(**default_kwargs)
        else:
            analyser = AnalysisFusionAcc(**fusion_analyser_kwargs)

    # glob paths
    m1_paths = natsorted(Path(m1_dir).glob("*"))
    m2_paths = natsorted(Path(m2_dir).glob("*"))
    fused_paths = natsorted(Path(fused_dir).glob("*"))
    print(f"Have {len(m1_paths)} images to analysis")
    assert (
        len(m1_paths) == len(m2_paths) == len(fused_paths)
    ), f"m1_paths, m2_paths and fused_paths should have the same number of images, but found {len(m1_paths)}, {len(m2_paths)}, {len(fused_paths)}"

    read_mode = ImageReadMode.RGB

    # import ipdb; ipdb.set_trace()
    if verbose:
        tbar = track(
            zip(m1_paths, m2_paths, fused_paths),
            description="Analysis ...",
            total=len(m1_paths),
            transient=True,
        )
    else:
        tbar = zip(m1_paths, m2_paths, fused_paths)

    for m1_path, m2_path, fused_path in tbar:
        m1_image = read_image(str(m1_path), mode=read_mode).float() / 255.0
        m2_image = read_image(str(m2_path), mode=read_mode).float() / 255.0
        fused_image = read_image(str(fused_path), mode=read_mode).float() / 255.0

        m1_image = m1_image.cuda().unsqueeze(0)
        m2_image = m2_image.cuda().unsqueeze(0)
        fused_image = fused_image.cuda().unsqueeze(0)

        analyser((m1_image, m2_image), fused_image)

    if verbose:
        print(analyser.result_str(to_table=True))

    return analyser.acc_ave


if __name__ == "__main__":
    from torchvision.io import read_image

    torch.cuda.set_device(1)
    analysis_by_dir(
        "/Data4/cao/ZiHanCao/exps/panformer/task_datasets/DIF/data/hier_test/deg_heavy_1_orig_size/LLVIP/snow/m1_clean",
        "/Data4/cao/ZiHanCao/exps/panformer/task_datasets/DIF/data/hier_test/deg_heavy_1_orig_size/LLVIP/snow/m2_clean",
        "/Data4/cao/ZiHanCao/exps/panformer/results/DIF/DIF_Flow/LLVIP/snow",
        # '/Data4/cao/ZiHanCao/exps/panformer/results/DIF/DDFM/LLVIP/snow',
        fusion_analyser_kwargs={
            "test_metrics": [
                "MUSIQ",
            ]
        },
        verbose=True,
    )
