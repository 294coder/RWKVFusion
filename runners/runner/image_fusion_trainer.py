import inspect
import math
import os
from pathlib import Path
from typing import Callable, Iterable

import accelerate
import hydra
import torch
import torch.nn as nn
from accelerate import Accelerator
from accelerate.state import AcceleratorState
from accelerate.utils import (
    DataLoaderConfiguration,
    DummyOptim,
    DummyScheduler,
    GradientAccumulationPlugin,
    ProjectConfiguration,
    set_seed,
    synchronize_rng_state,
)
from accelerate.utils.dataclasses import RNGType
from omegaconf import DictConfig, OmegaConf
from torch import Tensor
from torchmetrics.aggregation import MeanMetric

from model.base_model import BaseModel
from task_datasets import DATASET_KEYS
from utils import (
    EMA,
    AnalysisPanAcc,
    BestMetricSaveChecker,
    NameSpace,
    NoneLogger,
    StepsCounter,
    TensorboardLogger,
    check_fusion_mask_inp,
    default,
    dict_to_namespace,
    exists,
    flatten_namespace_first_level,
    get_loss,
    get_optimizer,
    get_scheduler,
    get_spectral_image_ready,
    get_train_dataset,
    has_patch_merge_model,
    patch_merge_in_val_step,
)
from utils.loss_utils import ave_multi_rank_dict
from utils.metric_fusion import AnalysisFusionAcc, MetricsByTask
from utils.misc import is_main_process
from utils.progress_utils import EasyProgress


def get_dtype(accelerator: accelerate.Accelerator | None = None):
    if accelerator is None:
        mix_precision = AcceleratorState().mixed_precision
    else:
        mix_precision = accelerator.mixed_precision

    if mix_precision == "fp16":
        return torch.float16
    elif mix_precision == "bf16":
        return torch.bfloat16
    elif mix_precision in (
        None,
        "none",
        "no",
        "fp32",
    ):  # compatible with non-accelerate mix-precision
        return torch.float32
    else:
        raise ValueError(f"Invalid mix_precision: {mix_precision}")


def default_gt_form(
    batch: dict[str, Tensor], fusion_task: str
) -> Tensor | dict[str, Tensor]:
    if fusion_task == "sharpening":
        return {"gt": batch["gt"]}
    elif fusion_task == "fusion":
        # for DRMFFusion loss
        return {""}
    else:
        raise ValueError(f"Invalid fusion_task: {fusion_task}")


class ImageFusionTrainer:
    def __init__(
        self,
        cfg: DictConfig | NameSpace,
        criterion_gt_form_fn: Callable | None = None,
        batch_inp_check_fns: Callable | list[Callable] | None = None,
    ):
        if isinstance(cfg, DictConfig):
            self.cfg = dict_to_namespace(OmegaConf.to_container(cfg, resolve=True))
        self.train_cfg = self.cfg.train
        self.val_cfg = self.cfg.val
        self.model_cfg = self.cfg.model
        self.datasets_cfg = self.cfg.datasets

        # type hints
        self.train_cfg: NameSpace
        self.val_cfg: NameSpace
        self.model_cfg: NameSpace
        self.datasets_cfg: NameSpace

        self.accelerator: Accelerator = hydra.utils.instantiate(cfg.accelerator)
        self.device = self.accelerator.device

        # model variants are different in dataset
        model_used = self.cfg.model_used
        model_cfg = (
            self.model_cfg.network_cfg[model_used]
            if model_used is not None
            else self.model_cfg.network_cfg
        )
        self.model: nn.Module = hydra.utils.instantiate(
            self.model_cfg.network_cfg[model_used]
        )
        self.model = self.model.to(self.device)
        self.ema_model = EMA(
            self.model,
            decay=self.train_cfg.ema_decay,
            update_every=self.train_cfg.ema_update_every,
            update_after_step=self.train_cfg.ema_update_after_step,
            state_include_online_model=False,
        )

        # optimizer and lr_scheduler
        self.optimizer, self.lr_scheduler = self.prepare_optimizer_and_scheduler()

        # loss function
        loss_used = self.cfg.loss_used
        loss_cfg = (
            self.cfg.loss_cfg[loss_used] if loss_used is not None else self.cfg.loss_cfg
        )
        self.criterion = hydra.utils.instantiate(loss_cfg)
        if hasattr(self.criterion, "to"):
            self.criterion = self.criterion.to(self.device)
        self.gt_form_fn = default(criterion_gt_form_fn, default_gt_form)

        # dataset
        _args_for_ds = flatten_namespace_first_level(self.cfg)
        _, self.train_dl, _, self.val_dl = get_train_dataset(_args_for_ds)
        self.fusion_task = self.cfg.fusion_task = _args_for_ds.fusion_task
        assert self.fusion_task in (
            "sharpening",
            "fusion",
        ), 'fusion_task should be "sharpening" or "fusion"'

        # make dataloaders to lists
        self.train_dl: list
        self.val_dl: list

        # input checks
        self.batch_inp_check_fns = default(
            batch_inp_check_fns,
            check_fusion_mask_inp
            if self.fusion_task == "fusion"
            else lambda *args: args,
        )

        # forward train and val step mode
        _train_step_mode = f"{self.fusion_task}_train"
        _val_step_mode = f"{self.fusion_task}_eval"
        self.step_mode = {
            "train": _train_step_mode
            if hasattr(self.model, _train_step_mode)
            else "train",
            "val": _val_step_mode if hasattr(self.model, _val_step_mode) else "val",
        }

        # loss
        loss = self.train_cfg.loss

        # model is trained on y-channel
        self.model_is_y_pred = self.train_cfg.only_y_train

        # metrics
        if self.cfg.fusion_task == "sharpening":
            self.analysis_reduced = AnalysisPanAcc(
                ratio=self.train_cfg.default_getattr("pansharpening_factor", 4),
                ref=True,
                sensor=self.datasets_cfg.dataset,
            )
            self.analysis_full = AnalysisPanAcc(
                ratio=self.train_cfg.default_getattr("pansharpening_factor", 4),
                ref=False,
                sensor=self.datasets_cfg.dataset,
            )
        elif self.cfg.fusion_task == "fusion":
            self.analysis = AnalysisFusionAcc(
                test_metrics=MetricsByTask.ON_TRAIN, only_on_y_component=True
            )
        else:
            raise NotImplementedError(f"{self.cfg.fusion_task} is not implemented")

        # logger
        if self.is_main_process:
            self.logger = TensorboardLogger(self.train_cfg)
            self.logger.watch(
                network=self.model,
                watch_type=self.train_cfg.watch_type,
                freq=self.train_cfg.watch_log_freq,
            )
        else:
            self.logger = NoneLogger(self.train_cfg)

        # counter
        self.train_state_counter = StepsCounter(["train", "val"])

        # save checker
        self.save_checker: BestMetricSaveChecker = hydra.utils.instantiate(
            self.train_cfg.save_checker
        )

        # save path
        self.weights_dir, self.log_file_dir, self.ema_model_path = (
            self.prepare_save_paths()
        )

        # accelerate prepare
        self.prepare_for_train()
        self.dtype = get_dtype(self.accelerator)

        # seed and rng
        set_seed(default(self.train_cfg.seed, 2025), device_specific=True)
        synchronize_rng_state(RNGType.TORCH)

        # summary the trainer
        self.logger.info(
            f"=" * 40 + "\n",
            f"Trainer Summary: \n",
            f"\t fusion task: {self.fusion_task} \n",
            f"\t model: {type(self.model).__name__} \n",
            f"\t optimizer: {type(self.optimizer).__name__} \n",
            f"\t lr_scheduler: {type(self.lr_scheduler).__name__} \n",
            f"\t criterion: {type(self.criterion).__name__} \n",
            f"\t save checker: {self.save_checker} \n",
        )

    @property
    def is_main_process(self):
        return self.accelerator.is_main_process

    def get_progress(
        self,
        task_desp: str,
        n_iters: int,
    ):
        tbar, prog_task_id = EasyProgress.easy_progress(
            [task_desp],
            [n_iters],
            is_main_process=self.is_main_process,
            tbar_kwargs={"console": self.logger._console},
            debug=self.train_cfg.debug,
        )

        return tbar, prog_task_id

    def update_progress(
        self,
        prog_task_id: int,
        n: int,
        bar_descp: str = "",
        total: int | None = None,
        visible: bool = True,
        close: bool = False,
    ):
        if (
            self.is_main_process
            and close
            and prog_task_id is not None
            and not self.train_cfg.debug
        ):
            EasyProgress.close_task_id(prog_task_id)

        if (
            self.is_main_process
            and not self.train_cfg.debug
            and prog_task_id is not None
        ):
            tbar = EasyProgress.tbar

            tbar.update(
                prog_task_id,
                total=total,
                completed=n,
                visible=visible,
                description=bar_descp,
            )

    def prepare_save_paths(self):
        if self.is_main_process:
            log_file_dir = Path(self.logger.log_file_dir)
            weights_dir = self.log_file_dir / "weights"
            ema_model_path = self.weights_dir / "ema_model.safetensors"
            weights_dir.mkdir(exist_ok=True, parents=True)
        else:
            # for ddp
            weights_dir = log_file_dir = ema_model_path = [None]

        if self.accelerator.use_distributed:
            (weights_dir, log_file_dir, ema_model_path) = (
                accelerate.utils.broadcast_object_list(
                    [weights_dir, log_file_dir, ema_model_path]
                )
            )
        self.accelerator.project_configuration.project_dir = self.weights_dir
        self.logger.info(
            "network params and training states are saved at [dim green]{}[/dim green]".format(
                self.ema_model_path
            )
        )

        return weights_dir, log_file_dir, ema_model_path

    @property
    def global_train_step(self):
        return self.train_state_counter.get("train")

    @property
    def global_val_step(self):
        return self.train_state_counter.get("val")

    def prepare_optimizer_and_scheduler(self):
        accelerator = self.accelerator

        # Construct no-weight-decay params
        _default_no_decay_names = ["bias", "LayerNorm.weight"]
        no_decay_names = self.train_cfg.default_getattr(
            "no_decay_param_names", _default_no_decay_names
        )
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if not any(nd in n for nd in no_decay_names)
                ],
                "weight_decay": self.cfg.optimizer_cfg.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if any(nd in n for nd in no_decay_names)
                ],
                "weight_decay": 0.0,
            },
        ]

        # Creates Dummy Optimizer if `optimizer` was specified in the config file else creates Adam Optimizer
        if (
            accelerator.state.deepspeed_plugin is None
            or "optimizer" not in accelerator.state.deepspeed_plugin.deepspeed_config
        ):
            optimizer = get_optimizer(
                self.model,
                self.model.parameters(),
                **self.model_cfg.optimizer.to_dict(),
            )
        else:
            optimizer = DummyOptim(optimizer_grouped_parameters)

        # Creates Dummy Scheduler if `scheduler` was specified in the config file else creates `args.lr_scheduler_type` Scheduler
        if (
            accelerator.state.deepspeed_plugin is None
            or "scheduler" not in accelerator.state.deepspeed_plugin.deepspeed_config
        ):
            # transformers lr scheduler
            # from transformers import get_scheduler
            # lr_scheduler = get_scheduler(
            #     name=args.lr_scheduler_type,
            #     optimizer=optimizer,
            #     num_warmup_steps=args.num_warmup_steps,
            #     num_training_steps=args.max_train_steps,
            # )
            lr_scheduler = get_scheduler(
                optimizer, **self.model_cfg.lr_scheduler.to_dict()
            )
        else:
            lr_scheduler = DummyScheduler(optimizer)

        return optimizer, lr_scheduler

    def prepare_for_train(self):
        # dataloader preparation
        is_dataloader = lambda loader: isinstance(loader, torch.utils.data.DataLoader)
        self.model, self.optimizer, self.lr_scheduler = self.accelerator.prepare(
            self.model, self.optimizer, self.lr_scheduler
        )
        if is_dataloader(self.train_dl):
            self.train_dl = [
                self.accelerator.prepare(self.train_dl)
            ]  # force to be a list
        else:  # is list of dataloader
            self.train_dl = [self.accelerator.prepare(dl) for dl in self.train_dl]
        if is_dataloader(self.val_dl):
            self.val_dl = [self.accelerator.prepare(self.val_dl)]
        else:  # is list of dataloader
            self.val_dl = [self.accelerator.prepare(dl) for dl in self.val_dl]

        # ema and counter checkpoints
        self.accelerator.register_for_checkpointing(self.ema_model)
        self.accelerator.register_for_checkpointing(self.train_state_counter)

        self.logger.info("[Train]: prepare for training")

    def model_out_check(
        self,
        model_out: Tensor | tuple[Tensor, Tensor] | tuple[Tensor, tuple],
        batch: dict,
    ) -> tuple[Tensor, Tensor, dict[str, Tensor]]:
        def loss_check(loss):
            # handle loss
            if isinstance(loss, tuple):
                bkwd_loss, loss_dict = loss
            elif isinstance(loss, Tensor):
                bkwd_loss = loss
                loss_dict = {"main_loss": loss}
            else:
                raise ValueError("loss is not a tensor or tuple")

            return bkwd_loss, loss_dict

        if isinstance(model_out, (tuple, list)):
            out, loss = model_out
            return out, loss_check(loss)
        else:
            self.logger.warning(
                "model out is not a tuple or list, which not contains loss",
                "manully call criterion",
                "prefix `gt` in batch may cause Item not Found",
            )
            loss = self.criterion(
                out, batch["gt"]
            )  # prefix key, this line is not necessary
            return out, loss_check(loss)

    def model_input_check(self, batch: dict):
        if self.batch_inp_check_fns is None or len(self.batch_inp_check_fns) == 0:
            return batch

        for fn in self.batch_inp_check_fns:
            batch = fn(batch)

        return batch

    def log_metrics(
        self,
        metrics_or_losses: dict,
        step: int | None = None,
        prefix: str | None = None,
    ):
        step = default(step, self.global_train_step)
        self.logger.log_curves(metrics_or_losses, step, prefix=prefix)

    def log_images(self, batched_img_dict: dict[str, Tensor], step: int | None = None):
        step = default(step, self.global_train_step)
        log_imgs = []
        log_img_names = []

        for name, img_batched in batched_img_dict.items():
            grid_img = get_spectral_image_ready(
                img_batched,
                tensor_name=name,
                task=self.fusion_task,
                ds_name=self.train_cfg.dataset,
            )
            log_imgs.append(grid_img)
            log_img_names.append(name)

        self.logger.log_images(
            batch_imgs=log_imgs, step=step, img_names=log_img_names, nrow=4
        )
        self.accelerator.wait_for_everyone()

    def backward_loss_updates(self, loss: Tensor):
        # assert loss
        assert loss.numel() == 1, "loss should be a scalar tensor"

        self.optimizer.zero_grad()
        self.accelerator.backward(loss)

        # check nan gradient
        if self.accelerator.sync_gradients:
            for name, param in self.model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    self.logger.warning(
                        f"step {self.global_train_step} - {name} has nan gradient",
                        proc_id=self.accelerator.process_index,
                    )
                    torch.nan_to_num(
                        param.grad, nan=0.0, posinf=1e5, neginf=-1e5, out=param.grad
                    )

            if self.dtype != torch.float16:
                self.accelerator.clip_grad_norm(
                    self.model.parameters(), self.train_cfg.max_grad_norm
                )

        # across processes
        self.optimizer.step()

        # update ema
        if self.accelerator.sync_gradients:
            self.ema_model.update()
            self.lr_scheduler.step()

        # train step counter update
        self.update_step("train")

    def infinity_train_loader(
        self, batch_n: int | None = None, per_loader_sample: bool = False
    ):
        if batch_n is not None:
            loader_batch_n = (
                batch_n * len(self.train_dl) if per_loader_sample else batch_n
            )
        else:
            loader_batch_n = math.inf

        if isinstance(self.train_dl, torch.utils.data.DataLoader):
            while True:
                for batch in self.train_dl:
                    yield (batch, 0)
                    loader_batch_n -= 1
                    if loader_batch_n <= 0:
                        return
        elif isinstance(self.train_dl, Iterable):
            while True:
                for loader_idx, dl in enumerate(self.train_dl):
                    for batch in dl:
                        yield (batch, loader_idx)
                        loader_batch_n -= 1
                        if loader_batch_n <= 0:
                            return
        else:
            raise TypeError(
                f"train_dl should be DataLoader or Iterable, but got {type(self.train_dl)}"
            )

    def finity_val_loader(
        self, batch_n: int | None = None, per_loader_sample: bool = False
    ):
        if batch_n is not None:
            loader_batch_n = (
                batch_n * len(self.val_dl) if per_loader_sample else batch_n
            )
        else:
            loader_batch_n = math.inf

        if isinstance(self.val_dl, torch.utils.data.DataLoader):
            for batch in self.val_dl:
                yield (batch, 0)
                loader_batch_n -= 1
                if loader_batch_n <= 0:
                    return
        elif isinstance(self.val_dl, Iterable):
            for loader_idx, dl in enumerate(self.val_dl):
                for batch in dl:
                    yield (batch, loader_idx)
                    loader_batch_n -= 1
                    if loader_batch_n <= 0:
                        return
        else:
            raise TypeError(
                f"val_dl should be DataLoader or Iterable, but got {type(self.val_dl)}"
            )

    def train_step(
        self, batch: dict, loader_idx: int
    ) -> dict[str, Tensor | dict[str, Tensor]]:
        model: BaseModel = self.model

        out, loss = model(
            batch,
            cfg=NameSpace(fusion_task=self.fusion_task),
            mode=self.step_mode["train"],
            criterion=self.criterion,
        )
        bkwd_loss, loss_dict = self.model_out_check(out, batch)

        # backward and step
        self.backward_loss_updates(bkwd_loss)

        return dict(model_out=out, loss_dict=loss_dict)

    def val_step(
        self, batch: dict, loader_idx: int
    ) -> dict[str, Tensor | dict[str, Tensor]]:
        model: BaseModel = self.model

        # default model forward
        model_forward = lambda batch: model(
            self.model_input_check(batch),
            cfg=NameSpace(fusion_task=self.fusion_task),
            mode=self.step_mode["val"],
        )

        extra_val_mode = self.train_cfg.extra_val_mode

        if self.fusion_task == "sharpening":
            ...

        elif self.fusion_task == "fusion":
            ...

        # loss = self.criterion(out, **self.gt_form_fn(batch))

    def update_step(self, mode: str):
        self.train_state_counter(mode)

    def train_loop(self):
        _, train_prg_iter = self.get_progress("train", self.train_cfg.train_iters)

        for batch, loader_idx in self.infinity_train_loader():
            batch = self.model_input_check(batch)
            result = self.train_step(batch, loader_idx)
            self.update_step("train")

        # validation loop
        if self.global_train_step % self.train_cfg.val_interval == 0:
            _, val_prg_id = self.get_progress(
                "validation", sum([len(dl) for dl in self.val_dl])
            )
            result_val = self.val_loop(val_prg_id)

        # save state
        will_save = self.save_checker(result_val["metrics"])
        if self.global_train_step % self.train_cfg.checkpoint_interval == 0:
            self.save_state()
            if will_save:
                self.save_ema()

        if self.global_val_step % self.train_cfg.visualize_val_internal == 0:
            self.log_images

    def val_loop(self, val_prg_task_id: int):
        self.logger.info("[Validation]: start validation process")

        val_loss_mean = MeanMetric()
        val_loss_dict = []

        for batch, loader_idx in self.finity_val_loader():
            # step
            batch = self.model_input_check(batch)
            result = self.val_step(batch, loader_idx)
            self.update_step("val")

            # loss avg
            val_loss_mean(result["val_loss"])
            val_loss_dict.append(result["val_loss_dict"])

            # update progress
            self.update_progress(
                prog_task_id=val_prg_task_id,
                n=self.global_val_step,
                visible=True,
                bar_descp="val",
            )

        # remove the temporary progress bar
        self.update_progress(
            prog_task_id=val_prg_task_id,
            n=self.global_val_step,
            visible=False,
            close=True,
        )

        # log metrics
        if self.fusion_task == "sharpening":
            metric_reduced = self.analysis_reduced.acc_ave
            metric_full = self.analysis_full.acc_ave

            # across rank
            metric_reduced_lst = self.accelerator.gather_for_metrics(metric_reduced)
            metric_full_lst = self.accelerator.gather_for_metrics(metric_full)

            # average
            metric_reduced = ave_multi_rank_dict(
                metric_reduced_lst, one_rank_zero=False
            )
            metric_full = ave_multi_rank_dict(metric_full_lst, one_rank_zero=False)

            # log
            self.log_metrics(metric_reduced)
            self.log_metrics(metric_full)
        elif self.fusion_task == "fusion":
            metric = self.analysis.acc_ave
            metric_lst = self.accelerator.gather_for_metrics(metric)
            metric = ave_multi_rank_dict(metric_lst, one_rank_zero=False)
            self.log_metrics(metric)

        # reset metric analysis
        self.reset_anlysis()

    def reset_anlysis(self):
        if self.fusion_task == "sharpening":
            self.analysis_reduced.clear_history()
            self.analysis_full.clear_history()
        elif self.fusion_task == "fusion":
            self.analysis.clear()

    def sanity_check(self):
        if self.train_cfg.sanity_check is None:
            return

        mode, sanity_check_n = self.train_cfg.sanity_check.split("_")
        sanity_check_n = int(sanity_check_n)
        if sanity_check_n <= 0:
            return

        # is train mode
        if mode == "train":
            for batch, loader_idx in self.infinity_train_loader(sanity_check_n):
                self.train_step(batch, loader_idx)
        # is val mode
        elif mode == "val":
            for batch, loader_idx in self.finity_val_loader(sanity_check_n):
                self.val_step(batch, loader_idx)
        else:
            raise ValueError(f"{mode} is not supported")

    def load_ema_model(self):
        # load EMA model
        pretrain_model_path = self.train_cfg.pretrain_model_path
        if pretrain_model_path is None and self.train_cfg.resume_path is not None:
            return

        assert Path(
            pretrain_model_path
        ).exists(), f"{pretrain_model_path} does not exist"
        assert (ext := Path(pretrain_model_path).suffix) in (
            ".safetensors",
            ".pth",
        ), f"{pretrain_model_path} is not a .safetensors or .pth file"

        state_dict = accelerate.utils.load_state_dict(pretrain_model_path)["ema"]
        self.model.load_state_dict(
            state_dict, strict=self.train_cfg.pretrain_load_strict
        )
        self.ema_model = EMA(
            self.model,
            decay=self.train_cfg.ema_decay,
            update_every=self.train_cfg.ema_update_every,
            update_after_step=self.train_cfg.ema_update_after_step,
            state_include_online_model=False,
        )

        self.logger.info(f"[Load EMA]: load ema model from {pretrain_model_path}")

    def resume_training(self):
        if self.train_cfg.resume_path is None:
            return

        self.accelerator.load_state(self.train_cfg.resume_path)
        self.logger.info(f"[Resume]: resume training from {self.train_cfg.resume_path}")

    def save_state(self):
        self.accelerator.wait_for_everyone()
        self.accelerator.save_state(self.weights_dir)
        self.logger.info("[Save]: save training states")

    def save_ema(self):
        self.accelerator.wait_for_everyone()
        ema_model = self.accelerator.unwrap_model(self.ema_model).ema_model
        self.accelerator.save_model(ema_model, self.ema_model_path)
        self.logger.info(f"[Save]: save ema model at {self.ema_model_path}")

    def run(self):
        self.logger.info("[Run]: starting training process")

        # resume
        self.load_ema_model()
        self.resume_training()

        # sanity check
        self.sanity_check()

        # train loop
        self.train_loop()
        self.val_loop()
