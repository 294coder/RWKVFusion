import torch
from __future__ import annotations
from typing import Tuple, Union
from torch import Tensor
from torch.optim.optimizer import Optimizer, ParamsT


def exists(val):
    return val is not None


class Tiger(Optimizer):
    r"""Tiger Optimizer
     A PyTorch implementation of the Tiger optimizer for Automatic Mixed Precision based on
    https://github.com/lucidrains/lion-pytorch/blob/main/lion_pytorch/foreach.py
    """

    def __init__(
        self,
        params: ParamsT,
        lr: Union[float, Tensor] = 1e-3,
        beta: float = 0.965,
        weight_decay: float = 1e-2,
    ):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {lr}")
        if not 0.0 <= beta < 1.0:
            raise ValueError("Invalid beta parameter: {beta}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        defaults = dict(lr=lr, beta=beta, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: None = None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr, beta, weight_decay = group["lr"], group["beta"], group["weight_decay"]

            params = []
            grads = []
            exp_avgs = []

            for p in filter(lambda p: exists(p.grad), group["params"]):
                grad, state = p.grad, self.state[p]

                if len(state) == 0:
                    state["exp_avg"] = torch.zeros_like(p)

                exp_avg = state["exp_avg"]

                params.append(p)
                grads.append(grad)
                exp_avgs.append(exp_avg)

            torch._foreach_mul_(params, 1.0 - lr * weight_decay)
            torch._foreach_mul_(exp_avgs, beta)
            torch._foreach_add_(exp_avgs, grads, alpha=1.0 - beta)
            updates = [exp_avg.clone() for exp_avg in exp_avgs]
            torch._foreach_sign_(updates)
            torch._foreach_add_(params, updates, alpha=-lr)
            torch._foreach_add_(exp_avgs, grads, alpha=1.0 - beta)

        return loss
