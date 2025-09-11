import os
import os.path as osp
import sys
import warnings

import accelerate
import hydra
import torch.nn as nn
from accelerate import Accelerator
from accelerate.utils import (
    DataLoaderConfiguration,
    DummyOptim,
    DummyScheduler,
    GradientAccumulationPlugin,
    ProjectConfiguration,
    set_seed,
)
from accelerate_run_fusion_engine import train as train_fusion
from accelerate_run_hyper_ms_engine import train as train_sharpening

from model import build_network
from utils import (
    BestMetricSaveChecker,
    EasyProgress,
    LoguruLogger,
    NameSpace,
    TensorboardLogger,
    config_load,
    convert_config_dict,
    get_loss,
    get_main_args,
    get_optimizer,
    get_scheduler,
    # get_fusion_dataset,
    get_train_dataset,
    is_main_process,
    merge_args_namespace,
)


def get_accelerate(args):
    accelerator = accelerate.Accelerator(
        mixed_precision="no",
        dataloader_config=DataLoaderConfiguration(
            split_batches=False,
            even_batches=False,
            non_blocking=True,
            dispatch_batches=None,
            use_seedable_sampler=False,
            use_stateful_dataloader=False,
        ),
        project_config=ProjectConfiguration(
            project_dir=args.proj_root,
            total_limit=args.ckpt_max_limit,
            automatic_checkpoint_naming=True,
        ),
        gradient_accumulation_plugin=GradientAccumulationPlugin(
            num_steps=args.grad_accum_steps,
            adjust_scheduler=False,
            sync_with_dataloader=False,
            sync_each_batch=False,
        ),
    )

    return accelerator


def get_network_config(args):
    ## old network config loading
    assert args.config_file is not None, "config_file must be specified"

    # get network config
    if args.config_file is not None:
        if not args.config_file.endswith(".yaml"):
            args.config_file += ".yaml"
        configs = config_load(args.config_file, end_with="")
    else:
        # old config loading
        configs = config_load(args.arch, "./configs")
    args = merge_args_namespace(args, convert_config_dict(configs))

    return args


def instantiate_network(args):
    # define network
    if args.model_class is not None:
        model_cls_module = args.network_configs.model_cls_module
        model_cls = hydra.utils.get_class(model_cls_module)
        network = model_cls(**vars(args.network_configs[args.model_class])).to(
            args.device
        )
        args.full_arch = args.model_class

    # may decrepted for old network registration
    else:
        warnings.warn(
            "initializing network from arch and sub_arch is deprecated, use model_class instead",
            DeprecationWarning,
        )
        full_arch = (
            args.arch + "_" + args.sub_arch if args.sub_arch is not None else args.arch
        )
        args.full_arch = full_arch
        network_configs = getattr(
            args.network_configs, full_arch, args.network_configs
        ).to_dict()
        network = build_network(full_arch, **network_configs).to(args.device)

    return network, args


def get_logger(args, accelerator: Accelerator, network: nn.Module):
    if args.logger_on and accelerator.is_main_process:
        logger = TensorboardLogger(
            tsb_comment=args.run_id,
            args=args,
            file_stream_log=True,
            method_dataset_as_prepos=True,
        )
        logger.watch(
            network=network,
            watch_type=args.watch_type,
            freq=args.watch_log_freq,
        )
    else:
        from utils import NoneLogger

        logger = NoneLogger(cfg=args, name=args.proj_name)

    return logger


def get_dataloader(args):
    if args.train_bs is None:
        args.train_bs = args.batch_size
    if args.val_bs is None:
        args.val_bs = args.batch_size

    # get train and val dataset and dataloader
    (_, train_dl, _, val_dl), args = get_train_dataset(args)

    return train_dl, val_dl, args


def make_save_path(args, accelerator, logger):
    if accelerator.is_main_process:
        weight_path = osp.join(logger.log_file_dir, "weights")
        os.makedirs(weight_path, exist_ok=True)
        args.output_dir = weight_path
        args.save_model_path = osp.join(args.output_dir, "ema_model.pth")
    else:
        # for ddp broadcast
        args.output_dir = args.save_model_path = args.save_base_path = [None]

    if accelerator.use_distributed:
        (args.output_dir, args.save_model_path, args.save_base_path) = (
            accelerate.utils.broadcast_object_list(
                [args.output_dir, args.save_model_path, args.save_base_path]
            )
        )
    accelerator.project_configuration.project_dir = args.output_dir
    logger.info(f"model weights will be save at {args.output_dir}")

    return args


def get_optimizer_and_scheduler(
    accelerator, network: nn.Module, logger, args: NameSpace
):
    # handle the optimizer and lr_scheduler
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in network.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": args.optimizer.weight_decay,
        },
        {
            "params": [
                p
                for n, p in network.named_parameters()
                if any(nd in n for nd in no_decay)
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
            network, network.parameters(), **args.optimizer.to_dict()
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
        lr_scheduler = get_scheduler(optimizer, **args.lr_scheduler.to_dict())
    else:
        # deepspeed lr_scheduler placeholder
        lr_scheduler = DummyScheduler(optimizer)
        #   total_num_steps=args.num_train_epochs,
        #   warmup_num_steps=args.warm_up_epochs)

    logger.info(
        "network params and training states are saved at [dim green]{}[/dim green]".format(
            args.save_model_path
        )
    )

    return optimizer, lr_scheduler


def get_criterion(args, network):
    loss_cfg = getattr(
        args, "loss_cfg", NameSpace()
    )  # TODO: make `loss_cfg` to `loss_configs`
    loss_cfg.fusion_model = network
    loss_cfg.device = args.device
    loss_cfg.model_is_y_pred = args.only_y_train
    criterion = get_loss(args.loss, **vars(loss_cfg[args.loss])).to(args.device)

    return criterion


def get_save_checker(args):
    if not args.regardless_metrics_save:
        metric_name = args.metric_name_for_save[args.task]
        check_order = args.metric_name_for_save.check_order[args.task]
        save_checker = BestMetricSaveChecker(
            metric_name=(
                vars(metric_name) if isinstance(metric_name, NameSpace) else metric_name
            ),
            check_order=check_order,
        )
    else:
        save_checker = None

    return save_checker


def main(args):
    accelerator = get_accelerate(args)
    set_seed(args.seed, device_specific=True)
    print(f">>> PID - {os.getpid()}: accelerate launching...")

    # set useful attributes
    device = accelerator.device
    args.device = str(device)
    args.ddp = accelerator.use_distributed
    args.world_size = accelerator.num_processes

    # get network config
    args = get_network_config(args)

    # instantiate network
    network, args = instantiate_network(args)

    # get logger
    logger = get_logger(args, accelerator, network)

    # make save path legal
    args = make_save_path(args, accelerator, logger)

    # get train dataset
    train_dl, val_dl, args = get_dataloader(args)

    # get loss
    criterion = get_criterion(args, network)

    # get optimizer and lr_scheduler
    optimizer, lr_scheduler = get_optimizer_and_scheduler(
        accelerator, network, logger, args
    )

    # save checker and train process tracker
    save_checker = get_save_checker(args)

    # * ==========================================================
    # * start train !

    if args.task == "fusion":
        train_fn = train_fusion
    elif args.task == "sharpening":
        train_fn = train_sharpening
    else:
        raise NotImplementedError(f"Task {args.task} is not implemented")

    train_fn(
        accelerator,
        network,
        optimizer,
        criterion,
        lr_scheduler,
        train_dl,
        val_dl,
        args.num_train_epochs,
        args.val_n_epoch,
        args.save_model_path,
        logger=logger,
        resume_epochs=1,
        check_save_fn=save_checker,
        args=args,
    )

    # logger finish
    if is_main_process() and logger is not None:
        logger.writer.close()


if __name__ == "__main__":
    logger = LoguruLogger.logger(sink=sys.stdout)
    LoguruLogger.add(
        "log_file/running_traceback.log",
        format="{time:MM-DD hh:mm:ss} {level} {message}",
        level="WARNING",
        backtrace=True,
        diagnose=True,
        mode="w",
    )
    LoguruLogger.add(
        sys.stderr,
        format="{time:MM-DD hh:mm:ss} {level} {message}",
        level="ERROR",
        backtrace=True,
        diagnose=True,
    )

    args = get_main_args()

    try:
        main(args)
    except Exception as e:
        if is_main_process():
            EasyProgress.close_all_tasks()
            logger.error(
                "An Error Occurred! Please check the stacks in log_file/running_traceback.log"
            )
            raise RuntimeError(f"Process {os.getpid()} encountered an error") from e
        else:
            raise RuntimeError(f"Process {os.getpid()} encountered an error") from e
