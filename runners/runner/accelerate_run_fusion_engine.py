import gc
import math
import os
import warnings
from typing import Callable, Union

import accelerate
import torch
from torch.utils.data import DataLoader

from utils.misc import NameSpace

# filter off all warnings
warnings.filterwarnings("ignore", module="torch")

from model.base_model import BaseModel
from task_datasets import DATASET_KEYS
from utils import (
    EMA,
    EasyProgress,
    TensorboardLogger,
    accum_loss_dict,
    ave_ep_loss,
    ave_multi_rank_dict,
    check_fusion_mask_inp,
    deep_speed_zero_n_init,
    deepspeed_to_device,
    default,
    default_none_getattr,
    dict_to_str,
    dist_gather_object,
    freeze_model,
    get_precision,
    loss_dict2str,
    module_load,
    pad_any,
    prefixed_dict_key,
    sanity_check,
    step_loss_backward,
    unfreeze_model,
    unpad_any,
    y_pred_model_colored,
)
from utils.metric_fusion import AnalysisFusionAcc, MetricsByTask
from utils.metric_sharpening import AnalysisPanAcc, NonAnalysis

# try to import
try:
    from schedulefree import AdamWScheduleFree
except ImportError:
    AdamWScheduleFree = None

from typing_extensions import TYPE_CHECKING

if TYPE_CHECKING:
    from accelerate.optimizer import AcceleratedOptimizer
    from nvidia.dali.plugin.pytorch import DALIGenericIterator
    from torch.utils.data import DataLoader

    from model.base_model import BaseModel
    from utils import BestMetricSaveChecker


def _optimizer_train(optimizer: "AcceleratedOptimizer"):
    if AdamWScheduleFree is None:
        return optimizer
    if isinstance(optimizer.optimizer, AdamWScheduleFree):
        optimizer.train()
    return optimizer


def _optimizer_eval(optimizer: "AcceleratedOptimizer"):
    if AdamWScheduleFree is None:
        return optimizer
    if isinstance(optimizer.optimizer, AdamWScheduleFree):
        optimizer.eval()
    return optimizer


def get_analyser(args):
    if args.log_metrics:
        # vis-ir fusion task
        if args.task == "fusion":
            analyser = AnalysisFusionAcc(
                only_on_y_component=True, test_metrics=MetricsByTask.ON_TRAIN
            )  # for fast evaluation
        else:
            analyser = AnalysisPanAcc(args.ergas_ratio)
    else:
        analyser = NonAnalysis()

    return analyser


def fuse_data_has_gt(data: dict, args):
    if hasattr(args, "has_gt"):
        args_gt = args.has_gt
    else:
        args_gt = True

    if "gt" in data:
        if data["gt"].size(1) == 3 and data["gt"].ndim == 4:
            data_gt = True
        else:
            data_gt = False
    else:
        data_gt = False

    return args_gt & data_gt


def get_save_checker(check_save_fn: "callable | BestMetricSaveChecker"):
    def _inner_check_fn(
        val_metrics: dict[str, float], val_loss: float | torch.Tensor = None, *args
    ):
        # set val_loss into val_metrics to check if save
        if val_loss is not None:
            val_metrics["val_loss"] = (
                val_loss.item() if torch.is_tensor(val_loss) else val_loss
            )
        return check_save_fn(val_metrics, *args)

    return _inner_check_fn


@torch.no_grad()
def val(
    accelerator: accelerate.Accelerator,
    network: BaseModel,
    val_dl: "DataLoader | DALIGenericIterator",
    criterion: Callable,
    logger: "TensorboardLogger",
    ep: int = None,
    optim_val_loss: float = None,
    args=None,
):
    val_loss = 0.0
    val_loss_dict = {}

    # get analysor for validate metrics
    analysis = get_analyser(args)

    dtype = get_precision(accelerator.mixed_precision)
    tbar, task_id = EasyProgress.easy_progress(
        ["Validation"],
        [len(val_dl)],
        is_main_process=accelerator.is_main_process,
        tbar_kwargs={"console": logger.console},
        debug=args.debug,
    )
    if accelerator.is_main_process and not args.debug:
        tbar.start()
    logger.print(
        "=======================================EVAL STAGE================================================="
    )
    modality_keys = DATASET_KEYS[args.fusion_task][:2]
    for i, data in enumerate(val_dl, 1):
        data = deepspeed_to_device(data, dtype=dtype)
        data = check_fusion_mask_inp(data, dtype=dtype)
        data, padder = pad_any(
            args, data, window_base=args.pad_window_base, pad_mode="resize"
        )
        with y_pred_model_colored(
            data, data_modality_keys=modality_keys, enable=args.only_y_train
        ) as (
            _data,
            back_to_rgb,
        ):
            outp = network(**_data, cfg=args, to_rgb_fn=back_to_rgb, mode="fusion_eval")

        if isinstance(outp, (tuple, list)):
            fused, _ = outp
        else:
            assert torch.is_tensor(outp), "output should be a tensor"
            fused = outp

        fused = padder.inverse(fused).clip(0, 1)
        _data = unpad_any(_data, padder, cfg=args)

        # import ipdb; ipdb.set_trace()
        gt = [_data[_modal] for _modal in modality_keys]
        loss_out = criterion(fused, gt, mask=getattr(_data, "mask", None))
        # if loss is hybrid, will return tensor loss and a dict
        if isinstance(loss_out, tuple):
            batched_loss, loss_d = loss_out
        else:
            batched_loss = loss_out
            loss_d = {"val_main_loss": loss_out}

        analysis(gt, fused)
        val_loss += batched_loss
        val_loss_dict = accum_loss_dict(val_loss_dict, loss_d)

        # advance the task_id
        if accelerator.is_main_process and task_id is not None and not args.debug:
            tbar.update(
                task_id,
                total=len(val_dl),
                completed=i,
                visible=True if i < len(val_dl) else False,
                description=f"Validation iter [{i}/{len(val_dl)}] - {loss_dict2str(loss_d)}",
            )

    logger.print(
        analysis.result_str(), dist=True, proc_id=accelerator.process_index
    )  # log in every process

    val_loss = accelerator.gather(val_loss)
    val_loss = val_loss.mean() / i
    val_loss_dict = ave_ep_loss(val_loss_dict, i)

    if args.log_metrics:  # gather from all procs to proc 0
        mp_analysis = dist_gather_object(analysis, n_ranks=accelerator.num_processes)
    gathered_val_dict = dist_gather_object(
        val_loss_dict, n_ranks=accelerator.num_processes
    )
    val_loss_dict = ave_multi_rank_dict(gathered_val_dict)

    # log validation results
    if args.log_metrics:
        if accelerator.is_main_process and args.ddp:
            for a in mp_analysis:
                logger.info(a.result_str())  # log in every process
        elif not args.ddp:
            logger.info(analysis.result_str())

    # gather metrics and log image
    acc_ave = analysis.acc_ave
    if accelerator.is_main_process:
        if args.ddp and args.log_metrics:
            n = 0
            acc = analysis.empty_acc
            for a in mp_analysis:
                for k, v in a.acc_ave.items():
                    acc[k] += v * a._call_n
                n += a._call_n
            for k, v in acc.items():
                acc[k] = v / n
            acc_ave = acc
        else:
            n = analysis._call_n

        # log validation curves
        if logger is not None:
            # metrics
            if args.log_metrics:
                logger.log_curves(
                    prefixed_dict_key(acc_ave, "val_metrics", sep="/"), ep
                )

            # losses
            for k, v in val_loss_dict.items():
                logger.log_curve(v, f"val_loss/{k}", ep)
            logger.log_curve(val_loss, "val_loss/loss", ep)

            # log validate image (last batch)
            modal1_name, modal2_name = DATASET_KEYS[args.fusion_task][:2]
            if data["gt"].shape[0] > 8:
                _plot_func = lambda x: x[:8]
                vi, ir, fused = list(
                    map(_plot_func, [data[modal1_name], data[modal2_name], fused])
                )
                if fuse_data_has_gt(data, args):
                    gt = _plot_func(data["gt"])
            else:
                vi, ir, gt = data[modal1_name], data[modal2_name], data["gt"]

            _logged_img = [vi, ir, fused]
            _logged_name = [modal1_name, modal2_name, "fused"]
            if fuse_data_has_gt(data, args):
                _logged_img.append(gt)
                _logged_name.append("gt")

            logger.log_images(
                _logged_img,
                nrow=4,
                names=_logged_name,
                epoch=ep,
                task=args.task,
                ds_name=args.dataset,
            )

        # print eval info
        logger.info("\n\nsummary of evaluation:")
        logger.info(f"evaluate {n} samples")
        logger.info(loss_dict2str(val_loss_dict))
        logger.info(f"\n{dict_to_str(acc_ave)}" if args.log_metrics else "")
        logger.info(
            "=================================================================================================="
        )

    # close task id
    if accelerator.is_main_process and task_id is not None and not args.debug:
        EasyProgress.close_task_id(task_id)
    # enf of validation
    accelerator.wait_for_everyone()

    return acc_ave, val_loss  # only rank 0 is reduced and other ranks are original data


def train(
    accelerator: accelerate.Accelerator,
    model: BaseModel,
    optimizer,
    criterion,
    lr_scheduler,
    train_dl: "DataLoader | DALIGenericIterator",
    val_dl: "DataLoader | DALIGenericIterator",
    epochs: int,
    eval_every_epochs: int,
    save_path: str,
    check_save_fn: "callable | BestMetricSaveChecker" = None,
    logger: "TensorboardLogger" = None,
    resume_epochs: int = 1,
    args: NameSpace = None,
):
    """
    Train the model using the provided parameters.

    Args:
        accelerator (accelerate.Accelerator): The Accelerator object for distributed training.
        model (BaseModel): The model to be trained.
        optimizer: The optimizer for training.
        criterion: The loss function.
        lr_scheduler: The learning rate scheduler.
        train_dl (DataLoader | DALIGenericIterator): The data loader for training data.
        val_dl (DataLoader | DALIGenericIterator): The data loader for validation data.
        epochs (int): The total number of training epochs.
        eval_every_epochs (int): The number of epochs between each evaluation.
        save_path (str): The path to save model checkpoints.
        check_save_fn (callable, optional): A function to determine if the model should be saved.
        logger (TensorboardLogger, optional): The logger for tracking training progress.
        resume_epochs (int): The epoch to resume training from.
        args: Additional arguments for training configuration.

    """
    # check save function
    # save_checker = lambda *check_args: check_save_fn(*check_args) if check_save_fn is not None else \
    #                lambda val_acc_dict, val_loss, optim_val_loss: val_loss < optim_val_loss
    save_checker = get_save_checker(check_save_fn)

    dtype = get_precision(accelerator.mixed_precision)

    # load pretrain model
    if args.pretrain_model_path is not None:
        # e.g., panMamba.pth
        model = module_load(
            args.pretrain_model_path,
            model,
            device=accelerator.device,
            strict=args.non_load_strict,
        )

    # Prepare everything with accelerator
    model, optimizer, train_dl, val_dl, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dl, val_dl, lr_scheduler
    )

    # FIXME: Deepspeed ZERO3 does not support EMA model
    # check if deepspeed is zero3
    if not deep_speed_zero_n_init(accelerator, n=[2, 3]):
        logger.info(f">>> use EMA model and register for checkpointing")
        ema_net = EMA(
            model,
            beta=args.ema_decay,
            update_every=args.default_getattr(
                "ema_update_every", 2 * accelerator.gradient_accumulation_steps
            ),
            update_after_step=args.default_getattr("ema_update_after_step", 100),
        )
        accelerator.register_for_checkpointing(ema_net)
    else:
        ema_net = None

    # load state
    if args.resume_path is not None:
        accelerator.load_state(input_dir=args.resume_path)
        logger.info(
            f">>> PROCESS {accelerator.process_index}: loaded state from {args.resume_path} done."
        )

    accelerator.wait_for_everyone()
    if args.sanity_check:
        logger.print(">>> sanity check...")
        model.eval()
        with torch.no_grad():
            freeze_model(model)
            sanity_check_val = sanity_check(val)
            sanity_check_val(
                accelerator, model, val_dl, criterion, logger, 0, torch.inf, args
            )
            unfreeze_model(model)
            torch.cuda.empty_cache()
            gc.collect()
        logger.print(f">>> sanity check done, ready to train the model")

    # Figure out how many steps we should save the Acclerator states
    # checkpointing_steps = args.checkpointing_steps
    # if checkpointing_steps is not None and checkpointing_steps.isdigit():
    #     checkpointing_steps = int(checkpointing_steps)

    optim_val_loss = math.inf
    fp_scaler = None

    logger.print(f">>> start training!")
    logger.print(f">>> Num Iterations per Epoch = {len(train_dl)}")
    logger.print(f">>> Num Epochs = {args.num_train_epochs}")
    logger.print(f">>> Gradient Accumulation steps = {args.grad_accum_steps}")

    optimizer = _optimizer_train(optimizer)

    # handle the progress bar
    tbar, (ep_task, iter_task) = EasyProgress.easy_progress(
        ["Epoch", "Iteration"],
        [epochs, len(train_dl)],
        is_main_process=accelerator.is_main_process,
        tbar_kwargs={"console": logger.console},
        debug=args.debug,
    )
    if accelerator.is_main_process and not args.debug:
        tbar.start()
    for ep in range(resume_epochs, epochs + 1):
        ep_loss = 0.0
        ep_loss_dict = {}
        i = 0
        _skip_n = 0

        # ======================================================== Model Training ===========================================================

        # model training
        for i, data in enumerate(train_dl, 1):
            data = deepspeed_to_device(data, dtype=dtype)
            data = check_fusion_mask_inp(data, dtype=dtype)

            # model get data and compute loss
            with accelerator.accumulate(model):
                modalities_key = DATASET_KEYS[args.fusion_task][:2]
                with y_pred_model_colored(
                    data, data_modality_keys=modalities_key, enable=args.only_y_train
                ) as (
                    data,
                    back_to_rgb,
                ):
                    data["fusion_criterion"] = criterion
                    data["has_gt"] = fuse_data_has_gt(data, args)
                    with accelerator.autocast():
                        _, loss_out = model(
                            **data, cfg=args, mode="fusion_train", to_rgb_fn=back_to_rgb
                        )

                # if loss is hybrid, will return tensor loss and a dict
                if isinstance(loss_out, (tuple, list)):
                    loss, loss_d = loss_out
                else:
                    assert isinstance(loss_out, torch.Tensor), "loss should be a tensor"
                    loss = loss_out

                # check nan loss
                if accelerator.gather(loss).isnan().any():
                    # clear the gradient and graph
                    del loss, loss_out, loss_d, data
                    model.zero_grad(set_to_none=True)
                    optimizer.zero_grad(set_to_none=True)
                    ema_net._zero_grad(set_to_none=True)
                    gc.collect()
                    # adam-mini optimizer will not be empty cached
                    torch.cuda.empty_cache()
                    # torch.cuda.ipc_collect()
                    torch.cuda.synchronize()
                    _skip_n += 1
                    logger.warning(
                        f">>> PROCESS {accelerator.process_index}: loss is nan, skip {_skip_n} batch(es) in this epoch"
                    )
                    continue

                # update parameters
                step_loss_backward(
                    optim=optimizer,
                    network=model,
                    max_norm=default_none_getattr(args, "max_norm"),
                    max_value=default_none_getattr(args, "max_value"),
                    loss=loss,
                    fp16=accelerator.mixed_precision != "fp32",
                    fp_scaler=fp_scaler,
                    accelerator=accelerator,
                    grad_accum=False,
                )

                ep_loss += loss
                if accelerator.sync_gradients and ema_net is not None:
                    ema_net.update()

            # update the progress bar
            ep_loss_dict = accum_loss_dict(ep_loss_dict, loss_d)
            if accelerator.is_main_process and not args.debug:
                tbar.update(
                    iter_task,
                    total=len(train_dl),
                    completed=i,
                    visible=True,
                    description=f"Training iter [{i}/{len(train_dl)}] - {loss_dict2str(loss_d)}",
                )

        # scheduler update
        # FIXME: not support transformers ReduceLROnPlateau which is LRLambda, may be using inspect can fix?
        if isinstance(lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            if not accelerator.optimizer_step_was_skipped:
                lr_scheduler.step(loss)
        else:
            if not accelerator.optimizer_step_was_skipped:
                lr_scheduler.step(ep)
            else:
                logger.info(
                    ">>> optimizer step was skipped due to mixed precision overflow"
                )

        # ======================================================== Validation ===========================================================
        # eval
        if (ep % eval_every_epochs == 0) and (eval_every_epochs != -1):
            optimizer = _optimizer_eval(optimizer)
            # eval model
            if ema_net is not None:
                ema_net.eval()
                eval_model = ema_net.ema_model
            else:
                model.eval()
                eval_model = model
            # close task id
            if accelerator.is_main_process and not args.debug:
                EasyProgress.close_task_id(iter_task)
                EasyProgress.close_task_id(ep_task)
            accelerator.wait_for_everyone()

            with accelerator.autocast():
                val_acc_dict, val_loss = val(
                    accelerator,
                    eval_model,
                    val_dl,
                    criterion,
                    logger,
                    ep,
                    optim_val_loss,
                    args,
                )
                torch.cuda.empty_cache()  # may be useful
                gc.collect()

            model.train()
            optimizer = _optimizer_train(optimizer)

            # ======================================================== Save Model ===========================================================
            # save ema model
            if not args.regardless_metrics_save:
                save_check_flag = save_checker(val_acc_dict, val_loss, optim_val_loss)
            else:
                save_check_flag = True

            if save_check_flag:
                if ema_net is not None:
                    saved_model = ema_net.ema_model
                else:
                    saved_model = model
                os.makedirs(os.path.dirname(save_path), exist_ok=True)

                # model state dict and optimizer state dict are saved separately
                accelerator.save_model(saved_model, save_path, safe_serialization=False)
                # accelerator.save_state(output_dir=save_path, safe_serialization=True)

                optim_val_loss = val_loss
                logger.print(
                    f">>> [green](validation)[/green] {ep=} - save params with best validation metrics"
                )

        # ======================================================== Checkpoint ===========================================================

        # checkpointing the running state
        checkpoint_flag = (
            False
            if args.checkpoint_every_n is None
            else ep % args.checkpoint_every_n == 0
        )
        if checkpoint_flag:
            logger.print(f">>> [red](checkpoint)[/red] {ep=} - save training state")
            output_dir = f"ep_{ep}"
            if args.output_dir is not None:
                output_dir = os.path.join(args.output_dir, output_dir)
            # save training state using accelerate config
            accelerator.save_state(output_dir, safe_serialization=False)

        # no validation, save ema model params per checkpoint_every_n if ema_net is not None (not deepspeed zero3)
        if (
            args.checkpoint_every_n is None
            and eval_every_epochs == -1
            and ema_net is not None
        ):
            logger.info(">>> [blue]EMA[/blue] - no validation, save ema model params")
            # save ema model and optimizer state dict
            accelerator.save_model(
                ema_net.ema_model, save_path, safe_serialization=False
            )
            accelerator.save_state(output_dir=save_path, safe_serialization=False)

        # ======================================================== Print Info ===========================================================

        accelerator.wait_for_everyone()

        # ep_loss average
        n_bwd = i - _skip_n
        assert n_bwd > 0, "no backward step in this epoch"
        ep_loss /= n_bwd
        ep_loss_dict = ave_ep_loss(ep_loss_dict, n_bwd)
        lr = optimizer.param_groups[0]["lr"]
        if accelerator.use_distributed:
            ep_loss = accelerator.reduce(ep_loss, reduction="mean")  # sum
            ep_loss_dict = dist_gather_object(
                ep_loss_dict, n_ranks=accelerator.num_processes, dest=0
            )  # gather n_proc objs
            ep_loss_dict = ave_multi_rank_dict(
                ep_loss_dict
            )  # [{'l1': 0.1}, {'l1': 0.2}] -> {'l1': 0.15}

        # advance the progress bar
        if accelerator.is_main_process:
            tbar.reset(iter_task)
            tbar.update(
                ep_task,
                total=epochs,
                completed=ep,
                visible=True,
                description=f"Epoch [{ep}/{epochs}] - ep_loss: {loss_dict2str(ep_loss_dict)}",
            )

        # print all info
        if logger is not None and accelerator.use_distributed:
            if accelerator.is_main_process:
                logger.log_curve(ep_loss, "train/loss", ep)
                for k, v in ep_loss_dict.items():
                    logger.log_curve(v, f"train/{k}", ep)
                logger.log_curve(lr, "train/lr", ep)
                logger.print(
                    f"[{ep}/{epochs}] lr: {lr:.4e} " + loss_dict2str(ep_loss_dict)
                )
        elif logger is None and accelerator.use_distributed:
            if accelerator.is_main_process:
                print(f"[{ep}/{epochs}] lr: {lr:.4e} " + loss_dict2str(ep_loss_dict))
        elif logger is not None and not accelerator.use_distributed:
            logger.log_curve(ep_loss, "train/loss", ep)
            for k, v in ep_loss_dict.items():
                logger.log_curve(v, f"train/{k}", ep)
            logger.log_curve(lr, "train/lr", ep)
            logger.print(f"[{ep}/{epochs}] lr: {lr:.4e} " + loss_dict2str(ep_loss_dict))
        else:
            print(f"[{ep}/{epochs}] lr: {lr:.4e} " + loss_dict2str(ep_loss_dict))

        # watch network params(grad or data or both)
        if isinstance(logger, TensorboardLogger):
            logger.log_network(model, ep)
