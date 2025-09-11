from functools import partial
from typing import TYPE_CHECKING, Tuple, Union
from warnings import warn

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from utils.misc import default_getattr

from .module import PatchMergeModule

if TYPE_CHECKING:
    from utils.misc import NameSpace


## TODO: old import fasion, now we directly import from the model path
# register all model name in a global dict
MODELS = {}


# use it in a decorator way
# e.g.
# @register_model('model_name')
def register_model(name):
    def inner(cls):
        MODELS[name] = cls
        return cls

    return inner


# base model class
# all model defination should inherit this class
from abc import ABC, abstractmethod


class BaseModel(ABC, nn.Module):
    @classmethod
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if not (
            cls._is_method_implemented("train_step")
            and cls._is_method_implemented("val_step")
        ):
            if not (
                cls._is_method_implemented("sharpening_train_step")
                or cls._is_method_implemented("fusion_train_step")
            ):
                raise NotImplementedError(
                    f"{cls.__name__} must implement at least one of the methods: 'sharpening_train_step' or 'fusion_train_step'"
                )

            if not (
                cls._is_method_implemented("sharpening_val_step")
                or cls._is_method_implemented("fusion_val_step")
            ):
                raise NotImplementedError(
                    f"{cls.__name__} must implement at least one of the methods: 'sharpening_val_step' or 'fusion_val_step'"
                )

    @staticmethod
    def _is_method_implemented(method):
        return any(method in B.__dict__ for B in BaseModel.__subclasses__())

    def train_step(
        self, ms, lms, pan, gt, criterion
    ) -> tuple[torch.Tensor, tuple[Tensor, dict[str, Tensor]]]:
        raise NotImplementedError

    def val_step(self, ms, lms, pan) -> torch.Tensor:
        raise NotImplementedError

    def fusion_train_step(
        self, vis, ir, mask, gt, criterion
    ) -> tuple[torch.Tensor, tuple[Tensor, dict[str, Tensor]]]:
        raise NotImplementedError

    def fusion_val_step(self, vis, ir, mask) -> torch.Tensor:
        raise NotImplementedError

    def sharpening_train_step(self, *args, **kwargs):
        raise NotImplementedError

    def sharpening_val_step(self, *args, **kwargs):
        raise NotImplementedError

    def patch_merge_step(self, *args) -> torch.Tensor:
        # not compulsory
        raise NotImplementedError

    def reorganize_inputs(self, *args, fusion_task: str = None, **kwargs):
        """
        reorganize the input args and kwargs according to the fusion task

        only used in training

        we leave the tuple args
        the train script is not used this tuple
        """

        if fusion_task is None:
            return args, kwargs

        if fusion_task == "VIF":
            ...
        elif fusion_task == "Pansharpening":
            ...
        elif fusion_task == "HMIF":
            kwargs["pan"] = kwargs.pop("rgb")
            kwargs["lms"] = kwargs.pop("hsi_up")
        elif fusion_task == "MEF":
            kwargs["vi"] = kwargs.pop("over")
            kwargs["ir"] = kwargs.pop("under")
        elif fusion_task == "MFF":
            kwargs["vi"] = kwargs.pop("far")
            kwargs["ir"] = kwargs.pop("near")
        elif fusion_task == "medical_fusion":
            kwargs["vi"] = kwargs.pop("s1")
            kwargs["ir"] = kwargs.pop("s2")
        else:
            raise ValueError(f"Invalid fusion_task: {fusion_task}")

        return args, kwargs

    def forward(self, *args, cfg=None, mode="train", **kwargs):
        if (
            mode == "sharpening_train"
            and not hasattr(self, "sharpening_train_step")
            or mode == "fusion_train"
            and not hasattr(self, "fusion_train_step")
        ):
            mode = "train"
        if (
            mode == "sharpening_eval"
            and not hasattr(self, "sharpening_val_step")
            or (mode == "fusion_eval" and not hasattr(self, "fusion_val_step"))
        ):
            mode = "eval"

        # reorganize the input args
        args, kwargs = self.reorganize_inputs(
            *args, fusion_task=default_getattr(cfg, "fusion_task", None), **kwargs
        )

        # forward the model
        if mode == "train":
            return self.train_step(*args, **kwargs)
        elif mode == "eval":
            return self.val_step(*args, **kwargs)
        elif mode == "sharpening_train":
            return self.sharpening_train_step(*args, **kwargs)
        elif mode == "sharpening_eval":
            return self.sharpening_val_step(*args, **kwargs)
        elif mode == "fusion_train":
            return self.fusion_train_step(*args, **kwargs)
        elif mode == "fusion_eval":
            return self.fusion_val_step(*args, **kwargs)
        elif mode == "patch_merge":
            warn("patch_merge is deprecated.", DeprecationWarning)
            # return self.patch_merge_step(*args, **kwargs)
        else:
            raise NotImplementedError(f"not implemented for mode: {mode}")

    def _forward_implem(self, *args, **kwargs):
        raise NotImplementedError
