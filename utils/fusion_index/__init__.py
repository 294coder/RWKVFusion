import warnings

import torch

from ._metric_MEF_MFF import evaluate_MEF_MFF_metric_torch
from ._old_VIF_metric import analysis_Reference_fast as evaluate_VIF_metric_numpy_old
from .metric_VIS_IR import evaluate_VIF_metric_numpy, evaluate_VIF_metric_torch


class FusionIndexWarning(UserWarning):
    pass


def evaludate_fusion_metrics(
    img_A: torch.Tensor,
    img_B: torch.Tensor,
    img_fused: torch.Tensor,
    metrics: "list[str] | str" = "all",
) -> dict[str, float]:
    # warn once
    warnings.simplefilter(action="once", category=FusionIndexWarning)
    warnings.warn("This function is deprecated and will be removed in the future.")

    if img_A.shape != img_B.shape or img_A.shape != img_fused.shape:
        raise ValueError("img_A, img_B, img_fused must have the same shape")

    if img_A.ndim != 2:
        raise ValueError("img_A, img_B, img_fused must be 2D tensor")

    VIF_metrics = evaluate_VIF_metric_torch(
        image_f=img_fused, image_ir=img_A, image_vis=img_B, metrics=metrics
    )

    MEF_MFF_metrics = evaluate_MEF_MFF_metric_torch(
        img_A=img_A, img_B=img_B, img_fused=img_fused, metrics=metrics
    )

    metrics = {**VIF_metrics, **MEF_MFF_metrics}

    return metrics


__all__ = [
    "evaludate_fusion_metrics",
    "evaluate_MEF_MFF_metric_torch",
    "evaluate_VIF_metric_torch",
    "evaluate_VIF_metric_numpy",
    "evaluate_VIF_metric_numpy_old",
]
