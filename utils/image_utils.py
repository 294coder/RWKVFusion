import warnings
from functools import reduce
from typing import Sequence

import kornia.augmentation as K
import torch
import torchvision.transforms.functional as T
from kornia.constants import Resample
from omegaconf import DictConfig

from utils.misc import NameSpace, default

# dataset output keys with fusion task specified
DATASET_KEYS = {
    "pansharpening": ["ms", "lms", "pan", "gt", "txt"],
    "HMIF": ["rgb", "lr_hsi", "hsi_up", "gt", "txt"],
    "VIF": ["vi", "ir", "mask", "gt", "txt", "name"],
    "MEF": ["over", "under", "mask", "gt", "txt", "name"],
    "MFF": ["far", "near", "mask", "gt", "txt", "name"],
    "medical_fusion": ["s1", "s2", "mask", "gt", "txt", "name"],
}


class TVResizer:
    def __init__(self, resize_sz, last_ori_size):
        self._resize_sz = resize_sz
        self._last_img_ori_size = last_ori_size

    def __call__(self, img):
        return T.resize(
            img,
            size=self._resize_sz,
            interpolation=T.InterpolationMode.NEAREST_EXACT,  # same as align_corners=True
            antialias=False,
        )

    def inverse(self, img):
        return T.resize(
            img,
            size=self._last_img_ori_size,
            interpolation=T.InterpolationMode.NEAREST_EXACT,
            antialias=False,
        )


class WindowBasedPadder:
    _instance: "WindowBasedPadder | None" = None
    mode: str = "pad"
    padding_fn: "K.PadTo | K.Resize | None" = None
    pad_mode: str = "reflect"

    def __new__(cls, window_size=64, mode="pad"):
        if cls._instance is None:
            cls._instance = super(WindowBasedPadder, cls).__new__(cls)
            cls._instance.window_size = window_size
            cls._instance.mode = mode
            assert mode in (
                "pad",
                "resize",
                "resize_tv",
            ), "only support mode `pad`, `resize`, and `resize_tv`"
            cls._instance.padding_fn = None

        return cls._instance

    def find_least_size_pad(self, base_size: tuple, window_size: int):
        least_size = []
        for b_s in base_size:
            if b_s % window_size == 0:
                least_size.append(b_s)
            else:
                mult = b_s // window_size
                mult += 1
                least_size.append(mult * window_size)
        return least_size

    def __call__(
        self,
        img: "torch.Tensor | None",
        size: "Sequence[int] | None" = None,
        no_check_pad: bool = False,
        resample: Resample = None,
    ):
        if img is None:
            return None

        resample = default(resample, Resample.NEAREST)
        assert isinstance(
            resample, Resample
        ), "only support resample method in kornia.constants.Resample"

        if no_check_pad:
            assert self.padding_fn is not None
            return self.padding_fn(img)

        if size is not None:
            self._last_img_ori_size = size
            if self.mode == "pad":
                self.padding_fn = K.PadTo(size, keepdim=True, pad_mode=self.pad_mode)
            elif self.mode == "resize":
                warnings.warn(
                    "kornia resize align corners seems does not work well when using `nearest` resample, "
                    "we use `torchvision resize` by default."
                )
                self.padding_fn = K.Resize(
                    size,
                    keepdim=True,
                    resample=resample,
                    align_corners=True,
                    antialias=False,
                )
            elif self.mode == "resize_tv":
                self.padding_fn = TVResizer(size, self._last_img_ori_size)
        else:
            pad_size = self.find_least_size_pad(img.shape[-2:], self.window_size)
            self._last_img_ori_size = img.shape[-2:]
            if self.mode == "pad":
                self.padding_fn = K.PadTo(
                    pad_size, keepdim=True, pad_mode=self.pad_mode
                )
            elif self.mode == "resize":
                self.padding_fn = K.Resize(
                    pad_size, keepdim=True, resample=resample, align_corners=True
                )
            elif self.mode == "resize_tv":
                self.padding_fn = TVResizer(pad_size, self._last_img_ori_size)

        return self.padding_fn(img)

    def inverse(self, img: "torch.Tensor | None", resample: Resample = None):
        if img is None:
            return None

        resample = default(resample, Resample.NEAREST)
        assert isinstance(
            resample, Resample
        ), "only support resample method in kornia.constants.Resample"

        if self.mode == "pad":
            return self.padding_fn.inverse(img, size=self._last_img_ori_size)
        elif self.mode == "resize":
            return self.padding_fn.inverse(
                img, size=self._last_img_ori_size, resample=resample
            )
        elif self.mode == "resize_tv":
            self.padding_fn: TVResizer
            return self.padding_fn.inverse(img)


class NonPadder:
    def __init__(self, **kwargs):
        pass

    def inverse(self, data, **kwargs):
        return data


def pad_any(
    cfg: "NameSpace | DictConfig",
    tensors: dict[str, torch.Tensor],
    window_base: int = 56,
    keys_to_pad: "list[str] | None" = None,
    pad_mode: str = "pad",
):
    if window_base <= 0:
        return tensors, NonPadder()

    padder = WindowBasedPadder(window_base, mode=pad_mode)

    assert cfg.fusion_task in (
        "VIF",
        "MEF",
        "MFF",
        "medical_fusion",
        "eval_fusion",
    ), "Only support VIF, MEF, MFF, medical_fusion"

    def _pad_with_names(tensors, names: "list[str]"):
        no_check_pads = [False] * len(names)  # no-cache the padding_fn, for mask input
        for name, no_check_pad in zip(names, no_check_pads):
            if name in tensors:
                tensors[name] = padder(
                    tensors[name],
                    no_check_pad=no_check_pad,
                    resample=Resample.NEAREST if name == "mask" else None,
                )

    if keys_to_pad is None:
        keys_to_pad = DATASET_KEYS[cfg.fusion_task]
        keys_to_pad = keys_to_pad[: keys_to_pad.index("txt")]

    _pad_with_names(tensors, keys_to_pad)

    return tensors, padder


def unpad_any(
    tensors: dict[str, torch.Tensor],
    padder: WindowBasedPadder | NonPadder,
    keys_to_unpad: "list[str] | None" = None,
    fusion_task: str | None = None,
    cfg: "NameSpace | DictConfig | None" = None,
):
    assert (
        fusion_task is not None or cfg is not None
    ), "fusion_task or cfg should be provided"

    if isinstance(padder, NonPadder):
        return tensors

    def _unpad_with_names(tensors, names: "list[str]"):
        for name in names:
            if name in tensors:
                t = padder.inverse(
                    tensors[name], resample=Resample.NEAREST if name == "mask" else None
                )
                if t is not None:
                    t = t.clip(0, 1)
                tensors[name] = t

    if keys_to_unpad is None:
        if cfg is not None:
            dataset_key = cfg.fusion_task
        else:
            dataset_key = fusion_task
        keys_to_unpad = DATASET_KEYS[dataset_key]
        keys_to_unpad = keys_to_unpad[: keys_to_unpad.index("txt")]
    else:
        assert isinstance(keys_to_unpad, list), "keys_to_unpad should be a list"

    _unpad_with_names(tensors, keys_to_unpad)

    return tensors


def normalize_img(image: torch.FloatTensor, norm_by_channel: bool = True):
    """
    Normalize the image tensor to [0, 1].
    """
    if image.dim() == 4 and norm_by_channel:
        reduce_fn = lambda x, op: reduce(x, "b c h w -> b 1 1 1", op)  # keep dims
        min_chw = reduce_fn(image, "min")
        max_chw = reduce_fn(image, "max")
        image.sub_(min_chw).div_(max_chw - min_chw)
    elif image.dim() == 3:
        image.sub_(image.min()).div_(image.max() - image.min())
    else:
        raise ValueError(f"image must be a 3D or 4D tensor, but got {image.dim()}D.")

    return image


if __name__ == "__main__":
    from omegaconf import OmegaConf

    cfg = OmegaConf.create(
        dict(
            fusion_task="VIF",
        )
    )
    data = dict(
        vi=torch.randn(1, 3, 100, 100).cuda(),
        ir=torch.randn(1, 3, 100, 100).cuda(),
        mask=torch.randint(0, 9, (1, 1, 100, 100)).float().cuda(),
        gt=torch.randn(1, 3, 100, 100).cuda(),
        txt=torch.randn(1, 100, 100).cuda(),
        name="IR_00000000.png",
    )
    data, padder = pad_any(cfg, data, window_base=64, pad_mode="resize")
    print(data["vi"].shape)
    print(data["ir"].shape)
    print(data["mask"].shape)

    data = unpad_any(data, padder, cfg=cfg)
    print(data["vi"].shape)
    print(data["ir"].shape)
    print(data["mask"].shape)
