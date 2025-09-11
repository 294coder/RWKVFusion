import sys
import warnings
from functools import partial
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from accelerate import DistributedType, PartialState
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
    LambdaLR,
    LRScheduler,
    MultiStepLR,
    ReduceLROnPlateau,
)

from utils.log_utils import easy_logger

logger = easy_logger(func_name="optim_utils")

# *==============================================================
# * learning rate schedulers
# *==============================================================


def cosine_scheduler(
    base_value, final_value, epochs, niter_per_ep, warmup_epochs=0, start_warmup_value=0
):
    """
    copy from DINO. manually set learning lr every iteration.
    note that there is only half epoch of cosine, which means learning rate will not
    go back to the original.
    :param base_value:
    :param final_value:
    :param epochs:
    :param niter_per_ep:
    :param warmup_epochs:
    :param start_warmup_value:
    :return:
    """
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = final_value + 0.5 * (base_value - final_value) * (
        1 + np.cos(np.pi * iters / len(iters))
    )

    schedule = np.concatenate((warmup_schedule, schedule))
    assert len(schedule) == epochs * niter_per_ep
    return schedule


class LinearWarmupScheduler(LRScheduler):
    def __init__(self, opt: optim.Optimizer, init_value, warmup_value, warmup_epochs):
        self.opt = opt
        self.init_value = init_value
        self.warmup_value = warmup_value
        self.warmup_epochs = warmup_epochs
        self.values = np.linspace(init_value, warmup_value, warmup_epochs)
        self.now_index = 0

    def step(self):
        self.opt.param_groups[0]["lr"] = self.values[self.now_index]
        self.now_index += 1


class CosineAnnealingWarmRestartsReduce(CosineAnnealingWarmRestarts):
    """
    Cosine annealing restart learning rate scheduler with reducing learning rate
    in a fixed ratio after each restart.

    Args:
        opt (optim.Optimizer): optimizer
        T_0 (int): number of epochs for the first restart
        T_mult (int, optional): factor to increase T_i after a restart. Defaults to 1.
        lr_mult (float, optional): learning rate multiplier after each restart. Defaults to 1.
        eta_min (float, optional): minimum learning rate. Defaults to 0.
        last_epoch (int, optional): index of the last epoch. Defaults to -1.
        warmup_epochs (int, optional): number of epochs for linear warmup. Defaults to 0.
    """

    def __init__(
        self,
        opt: optim.Optimizer,
        T_0: int,
        T_mult: int = 1,
        lr_mult: float = 1,
        eta_min: float = 0,
        last_epoch: int = -1,
        warmup_epochs: int = 0,
    ):
        self.opt = opt
        self.lr_mult = lr_mult
        self.warmup_epochs = warmup_epochs
        self._warmed_up = False if warmup_epochs > 0 else True
        super().__init__(opt, T_0, T_mult, eta_min, last_epoch)

    def step(self, ep: int = None):
        if (
            self.warmup_epochs > 0
            and self.T_cur <= self.warmup_epochs
            and not self._warmed_up
            and self.T_cur >= 0
        ):
            self._last_lr = []
            for i in range(len(self.optimizer.param_groups)):
                # from eta_min to base_lr
                _ratio = self.T_cur / self.warmup_epochs
                _curr_lr = _ratio * self.base_lrs[i] + (1 - _ratio) * self.eta_min
                self.optimizer.param_groups[i]["lr"] = _curr_lr
                self._last_lr.append(_curr_lr)
            self.T_cur += 1
        elif self.T_cur > self.warmup_epochs and not self._warmed_up:
            self._warmed_up = True
            self.T_cur = 0
        elif self._warmed_up and self.T_cur == self.T_i - 1 and self.last_epoch != 0:
            # reduce the base lr
            for i in range(len(self.base_lrs)):
                self.base_lrs[i] *= self.lr_mult
                self.base_lrs[i] = max(self.base_lrs[i], self.eta_min)
            # step the scheduler by super().step() in cosine annealing way
            super().step()
        elif self.T_cur < 0:
            self.T_cur = 0
            self._last_lr = [group["lr"] for group in self.optimizer.param_groups]
        else:
            # neither warmup nor reduce the base_lr,
            # step the scheduler by super().step() in cosine annealing way
            super().step()


def get_scheduler(optim, **kwargs):
    """
    get lr_scheduler or weight_decay_scheduler
    Args:
        optim: optimizer
        **kwargs: a dict containing type of scheduler and its arguments

    Returns: a scheduler

    """
    name = kwargs["name"]
    kwargs.pop("name")
    if name == "cos_anneal":
        return CosineAnnealingLR(optim, **kwargs)
    elif name == "cos_anneal_restart":
        return CosineAnnealingWarmRestarts(optim, **kwargs)
    elif name == "cos_anneal_restart_reduce":
        return CosineAnnealingWarmRestartsReduce(optim, **kwargs)
    elif name == "multi_step":
        return MultiStepLR(optim, **kwargs)
    elif name == "plateau":
        return ReduceLROnPlateau(optim, **kwargs)
    elif name in ["constant", "identity"]:
        return LambdaLR(optim, lr_lambda=lambda step: 1.0)
    else:
        raise NotImplementedError


def get_precision(mixed_precision):
    if mixed_precision == "fp32" or mixed_precision == "no":
        return torch.float32
    elif mixed_precision == "fp16":
        return torch.float16
    elif mixed_precision == "bf16":
        return torch.bfloat16
    else:
        raise ValueError(f"Invalid mixed precision value: {mixed_precision}")


# *==============================================================
# * optimizers
# *==============================================================


def get_optimizer(
    model: torch.nn.Module | None = None,
    params: "Iterable | dict | None" = None,
    **kwargs,
):
    name = kwargs["name"]
    kwargs.pop("name")
    assert (
        model is not None or params is not None
    ), "model or params should be provided by at least one"

    if name == "sgd":
        return optim.SGD(params, **kwargs)
    elif name == "adam":
        return optim.Adam(params, **kwargs)
    elif name == "adamw":
        return optim.AdamW(params, **kwargs)
    elif name == "muon":
        from utils.utils_modules.moun_adamw import Muon

        muon_params, adamw_params = Muon.clear_muon_adamw_params(
            model.named_parameters()
        )

        return Muon(muon_params=muon_params, adamw_params=adamw_params, **kwargs)

    elif name == "lion":
        from lion_pytorch import Lion

        betas = kwargs.pop("betas", (0.95, 0.98))
        return Lion(params, betas=betas, **kwargs)
    elif name == "tiger":
        from .utils_modules.tiger import Tiger

        # https://kexue.fm/archives/9512
        kwargs.pop("betas", None)
        return Tiger(params, beta=0.945, **kwargs)
    elif name == "fusedadam":
        import deepspeed

        return deepspeed.ops.adam.FusedAdam(params, **kwargs)
    elif name == "schedulefree_adam":
        import schedulefree

        return schedulefree.AdamWScheduleFree(params, **kwargs)
    elif name == "adam_mini":
        from utils.utils_modules import Adam_mini

        if model is not None:
            raise RuntimeError(
                "Adam_mini optimizer should initialized with named parameters"
            )
        return Adam_mini(model.named_parameters(), **kwargs)
    elif name == "adamw_8bit":
        from bitsandbytes.optim import AdamW8bit

        return AdamW8bit(params, **kwargs)
    elif name == "ademamix":
        from utils.utils_modules import AdEMAMix

        return AdEMAMix(params, **kwargs)
    # * preconditional optmizer ==========================================================
    elif name == "shampoo":
        from torch_optimizer import Shampoo

        return Shampoo(params, **kwargs)
    elif name == "shampoo_ddp":
        from utils.utils_modules.shampoo_optimizers.distributed_shampoo.distributed_shampoo import (
            AdamGraftingConfig,
            DDPShampooConfig,
            DistributedShampoo,
        )
        from utils.utils_modules.shampoo_optimizers.distributed_shampoo.utils.shampoo_ddp_distributor import (
            CommunicationDType,
        )

        warnings.warn(
            "Shampoo optimizer has not been tested yet, may cause nan or other unexpected errors.",
            UserWarning,
        )

        state = PartialState()

        if state.distributed_type == DistributedType.MULTI_GPU:
            distributed_adam_config = DDPShampooConfig(
                communication_dtype=CommunicationDType.FP32,
                num_trainers_per_group=state.num_processes,
                communicate_params=False,
            )
        elif state.distributed_type == DistributedType.NO:
            distributed_adam_config = None
        else:
            raise ValueError(
                f"Shampoo optimizer only supports DDP and NO distributed type, but got {state.distributed_type}"
            )

        opt = DistributedShampoo(
            params,
            lr=kwargs["lr"],
            betas=kwargs.pop("betas", (0.9, 0.999)),
            epsilon=kwargs.pop("eps", 1e-12),
            weight_decay=kwargs.pop("weight_decay", 1e-5),
            max_preconditioner_dim=8192,
            precondition_frequency=100,
            use_nesterov=kwargs.pop("use_nesterov", False),
            use_pytorch_compile=kwargs.pop("use_pytorch_compile", True),
            use_decoupled_weight_decay=True,
            grafting_config=AdamGraftingConfig(
                beta2=kwargs.pop("beta2", 0.999),
                epsilon=kwargs.pop("eps", 1e-12),
            ),
            distributed_config=distributed_adam_config,
        )
        opt.state_dict = partial(
            opt.distributed_state_dict, key_to_param=model.named_parameters()
        )
        opt.load_state_dict = partial(
            opt.load_distributed_state_dict, key_to_param=model.named_parameters()
        )

        return opt
    elif name.startswith("soap"):
        from utils.utils_modules import (
            # baseline
            SOAP,
            # other variants
            ForeachSOAP,
            PaLMForeachSOAP,
            PrecondScheduleForeachSOAP,
            PrecondSchedulePaLMForeachSOAP,
        )

        optimizers_dict = dict(
            soap=SOAP,
            soap_for_each=ForeachSOAP,
            soap_palm=PaLMForeachSOAP,
            # soap_sf_palm=SFPaLMForeachSOAP,
            soap_precond=PrecondScheduleForeachSOAP,
            soap_precond_palm=PrecondSchedulePaLMForeachSOAP,
            # soap_precond_sf_palm=PrecondScheduleSFPaLMSOAP,
        )
        optimizer_cls = optimizers_dict.get(name, None)
        if optimizer_cls is None:
            logger.error(
                f"[r]Error[/r]: SOAP optimizer {name} not implemented, only support {list(optimizers_dict.keys())}"
            )
            raise NotImplementedError(f"SOAP optimizer {name} not implemented")
        return optimizer_cls(params, **kwargs)
    elif name.startswith("psgd"):
        from utils.utils_modules import (
            ForeachDelayedPSGD,
            ForeachPSGDKron,
            ForeachPurePSGD,
        )

        optimizers_dict = dict(
            psgd=ForeachPSGDKron,
            psgd_pure=ForeachPurePSGD,
            psgd_delayed=ForeachDelayedPSGD,
        )
        optimizer_cls = optimizers_dict.get(name, None)
        if optimizer_cls is None:
            logger.critical(
                f"[r]Error[/r]: PSGD optimizer {name} not implemented, only support {list(optimizers_dict.keys())}"
            )
            raise NotImplementedError(f"PSGD optimizer {name} not implemented")
        return optimizer_cls(params, **kwargs)

    # * cautious type optimizers =========================================================
    elif name == "c_adamw":
        from utils.utils_modules import CautiousAdamW

        return CautiousAdamW(params, **kwargs)
    elif name == "c_lion":
        from utils.utils_modules import CautiousLion

        return CautiousLion(params, **kwargs)

    # * mars type optimizer ===============================================================
    elif name == "mars":
        from utils.utils_modules import MARS

        return MARS(params, **kwargs)
    else:
        raise NotImplementedError(f"optimizer {name} not implemented")
