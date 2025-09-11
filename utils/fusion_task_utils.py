import torch
from contextlib import contextmanager
from typing import Union, Tuple, Optional, Dict, Iterable
import copy
import kornia


######################## training and inference utilities ########################


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


@contextmanager
def y_pred_model_colored(
    modalities: Union[Tuple[torch.Tensor, ...], Dict[str, torch.Tensor]],
    data_modality_keys: Optional[Tuple[str, ...]] = None,
    enable: bool = True,
):
    """
    Context manager for handling YCbCr color space conversion during two modality image fusion.

    Args:
        modalities: Union[Tuple[torch.Tensor, ...], Dict[str, torch.Tensor]]:
            The modalities to be processed. This can be a tuple of torch.Tensors or a dictionary with string keys and torch.Tensor values.
        data_modality_keys: Optional[Tuple[str, ...]]:
            The keys of the modalities in the data dictionary. If None, the keys are assumed to be the same as the modalities.
        enable: bool: Enables or disables the YCbCr color space conversion.
            If True, the conversion is enabled; otherwise, the original color space is used.

    Yields:
    torch.Tensor: The Y channel of the image

    This context manager is responsible for:
    1. Converting RGB to YCbCr
    2. Extracting the Y channel for processing
    3. Converting the processed Y channel back to RGB
    """

    def one_color_fusion(modal_1, modal_2):
        if modal_1.size(1) != 1:
            y_cb_cr = kornia.color.rgb_to_ycbcr(modal_1)
        else:
            y_cb_cr = kornia.color.rgb_to_ycbcr(modal_2)

        cbcr = y_cb_cr[:, 1:]

        def back_to_rgb(pred_y):
            y_cb_cr = torch.cat([pred_y, cbcr], dim=1)
            return kornia.color.ycbcr_to_rgb(y_cb_cr).clip(0, 1)

        return back_to_rgb

    def two_color_fusion(modal_1, modal_2):
        _s1_y_cb_cr = kornia.color.rgb_to_ycbcr(modal_1)
        _s2_y_cb_cr = kornia.color.rgb_to_ycbcr(modal_2)

        cb_1 = _s1_y_cb_cr[:, 1:2]
        cb_2 = _s2_y_cb_cr[:, 1:2]

        cr_1 = _s1_y_cb_cr[:, 2:3]
        cr_2 = _s2_y_cb_cr[:, 2:3]

        def back_to_rgb(pred_y):
            tau = 0.5
            mid_1 = cb_1 * (cb_1 - tau).abs() + cb_2 * (cb_2 - tau).abs()
            mid_2 = (cb_1 - tau).abs() + (cb_2 - tau).abs()
            _mask = mid_2 == 0
            cb_fused = mid_1 / mid_2
            cb_fused[_mask] = tau

            mid_3 = cr_1 * (cr_1 - tau).abs() + cr_2 * (cr_2 - tau).abs()
            mid_4 = (cr_1 - tau).abs() + (cr_2 - tau).abs()
            _mask = mid_4 == 0
            cr_fused = mid_3 / mid_4
            cr_fused[_mask] = tau

            y_cb_cr = torch.cat([pred_y, cb_fused, cr_fused], dim=1)
            return kornia.color.ycbcr_to_rgb(y_cb_cr).clip(0, 1)

        return back_to_rgb

    _is_dict = isinstance(modalities, dict)

    if enable:
        if _is_dict:
            assert (
                data_modality_keys is not None
            ), "data_modality_keys should be provided when using dict"
            assert (
                len(data_modality_keys) == 2
            ), "data_modality_keys should be a 2-tuple"
            _modalities = [modalities[k] for k in data_modality_keys]
        elif isinstance(modalities, (list, tuple)):
            assert len(modalities) == 2, "modalities should be a 2-tuple"
            _modalities = modalities
        else:
            raise ValueError(f"Invalid modalities type: {type(modalities)}")

        _color_n_inps = 0
        _y = []

        for modal in _modalities:
            if modal.size(1) == 1:  # is gray image
                _y.append(modal)
            else:  # is colored image
                _color_n_inps += 1
                _y.append(kornia.color.rgb_to_y(modal)[:, 0:1])

        if _color_n_inps == 1:
            back_to_rgb = one_color_fusion(*_modalities)
        elif _color_n_inps == 2:
            back_to_rgb = two_color_fusion(*_modalities)
        else:
            raise ValueError(
                f"Invalid number of color inputs: {_color_n_inps}, has modalities shapes: {[m.shape for m in _modalities]}"
            )

        if _is_dict:
            _y = {k: _y[i] for i, k in enumerate(data_modality_keys)}
            cp_modalities = copy.deepcopy(modalities)
            cp_modalities.update(_y)
            y = cp_modalities
        else:
            y = _y

    else:
        y = modalities

        def back_to_rgb(pred_rgb):
            # assume we have clip the fused image in the model's forward
            return pred_rgb

    try:
        # Yield the Y channel for processing
        yield y, back_to_rgb

    finally:
        pass
