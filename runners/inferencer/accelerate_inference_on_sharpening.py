import argparse
import os
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Callable

import h5py
import hydra
import numpy as np
import torch
from accelerate import Accelerator
from accelerate.utils import DataLoaderConfiguration
from omegaconf import OmegaConf
from scipy.io import savemat
from torch.utils.data import DataLoader
from tqdm import tqdm

if TYPE_CHECKING:
    from model.base_model import BaseModel

from model import build_network
from utils import (
    NameSpace,
    dict_data_to_device_and_type,
    easy_logger,
    get_eval_dataset,
    has_patch_merge_model,
    module_load,
    patch_merge_in_val_step,
)
from utils.metric_sharpening import AnalysisPanAcc

logger = easy_logger(func_name="inference")


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-c", "--config_file", type=str, default=None, help="config file path"
    )
    parser.add_argument(
        "-m", "--model_class", type=str, default=None, help="model class name"
    )
    parser.add_argument("--arch", type=str, default=None)
    parser.add_argument("--sub_arch", type=str, default=None)
    parser.add_argument("--dataset", type=str, default="tno")
    parser.add_argument("--model_path", type=str, default="", required=True)
    parser.add_argument("--load_from_logs", type=bool, default=True)
    parser.add_argument("--val_bs", type=int, default=2)
    parser.add_argument("--full_res", default=False, action="store_true")
    parser.add_argument("--split_patch", default=False, action="store_true")
    parser.add_argument(
        "--patch_size", type=int, default=64, help="patch size for split"
    )
    parser.add_argument("--patch_size_list", type=list, default=[16, 32, 64, 64])
    parser.add_argument("--ergas_ratio", type=int, default=4)
    parser.add_argument("--crop_bs", type=int, default=6)
    parser.add_argument("--save_format", type=str, default="mat", choices=["mat", "h5"])
    parser.add_argument("--save_mat", default=False, action="store_true")
    parser.add_argument("--val_step_func", type=str, default="sharpening_val_step")
    args = parser.parse_args()

    # to omegaconf
    conf = OmegaConf.create(args.__dict__, flags={"allow_objects": True})

    # load network config
    conf, yaml_cfg = network_loading(args, conf)

    # dataset and save path config
    conf.dataset = conf.dataset.lower()
    conf.path = yaml_cfg.path

    return conf


def network_loading(args, conf=None):
    if conf is None:
        conf = args

    # load network config
    # old model loading
    if args.model_class is None:
        conf.full_arch = (
            args.arch + "_" + args.sub_arch if args.sub_arch is not None else args.arch
        )
        conf.network_configs_path = Path("configs") / f"{conf.arch}_config.yaml"

        # merge args and yaml_cfg
        yaml_cfg = OmegaConf.load(conf.network_configs_path)
        conf.merge_with(yaml_cfg)
        conf.network_configs = getattr(
            yaml_cfg["network_configs"], conf.full_arch, yaml_cfg["network_configs"]
        )
    # new model loading
    else:
        assert args.config_file is not None, "config file should be provided"
        yaml_cfg = OmegaConf.load(args.config_file)
        conf.full_arch = conf.model_class
        conf.model_init_kwargs = yaml_cfg.network_configs[conf.model_class]
        # merge args and yaml_cfg
        conf.merge_with(yaml_cfg)
        logger.info(
            f"initilize model {conf.model_class} using configure: {conf.network_configs}"
        )
    return conf, yaml_cfg


def get_hydra_configs(args):
    # load args
    hydra.initialize("configs/sharpening_infer_cfgs")
    args = hydra.compose(config_name="unref_config")
    args = NameSpace.dict_to_namespace(OmegaConf.to_container(args))

    # load network config
    args, yaml_cfg = network_loading(args)

    # dataset and save path config
    args.dataset = args.dataset.lower()
    args.path = yaml_cfg.path

    return args


# *==============================================================
# * reduced-resolution and full-resolution inference functions
# *==============================================================


@torch.no_grad()
def unref_for_loop(
    model,
    dl: DataLoader,
    cfg: "NameSpace | None" = None,
    *,
    device: "str | torch.device | None" = None,
    split_patch=False,
    lifting_full_resolution: bool = False,
    feature_callback: Callable | None = None,
    sensor: str = "DEFAULT",
    factor: int = 4,
    default_max_value: float | None = None,
    lifting_full_resolution_kwargs: dict = {},
    patch_merge_module_kwargs: dict = {},  # ergas_ratio=4, sensor="DEFAULT", default_max_value=None
):
    """
    Performs inference on a model using a DataLoader without reference images.

    Args:
        model: The model to perform inference on.
        dl: The DataLoader providing the input data.
        cfg: Optional configuration namespace.
        device: The device to run inference on (e.g., 'cuda', 'cpu').
        split_patch: Whether to split the input into patches for memory efficiency.
        feature_callback: An optional callback function to process features during inference.
        lifting_full_resolution_kwargs: Keyword arguments for the PansharpeningUnsupervisedLoss.
        patch_merge_module_kwargs: Keyword arguments for the PatchMergeModule.

    Returns:
        A list of super-resolved images as numpy arrays.
    """
    from model.base_model import PatchMergeModule

    all_sr = []
    inference_bar = tqdm(
        enumerate(dl, 1), dynamic_ncols=True, total=len(dl), desc="Inference"
    )

    # analyser
    analysis = AnalysisPanAcc(
        ratio=factor,
        ref=False,
        sensor=sensor,
        default_max_value=default_max_value,
    )

    # if using patch merge module
    _val_step_func = patch_merge_module_kwargs.pop("val_step_func", "val_step")
    if split_patch:
        # check if has the patch merge model
        if not (
            has_patch_merge_model(model)
            or patch_merge_in_val_step(model, val_step_func=_val_step_func)
        ):
            # assert bs == 1, 'batch size should be 1'
            # warp the model into PatchMergeModule
            model = PatchMergeModule(
                net=model, device=device, **patch_merge_module_kwargs
            )

    # if lifting full resolution module (using unsupervised pansharpening loss)
    if lifting_full_resolution:
        assert (
            len(lifting_full_resolution_kwargs) != 0
        ), "lifting_full_resolution_kwargs should not be empty when lifting full resolution else set to False"
        # TODO: implement lifting full resolution when split_patch is True
        assert (
            not split_patch
        ), 'split_patch should be False when using "lifting_full_resolution"'

    # * ==========================================================
    # * full-resoliution inference

    for i, data in inference_bar:
        data = dict_data_to_device_and_type(data, device, dtype=torch.float32)

        # split the image into several patches to avoid gpu OOM
        if split_patch:
            if hasattr(model, "forward_chop") or isinstance(model, PatchMergeModule):
                # is PatchMergeModule
                input = (data["ms"], data["lms"], data["pan"])
                sr = model.forward_chop(*input)[0]
            elif patch_merge_in_val_step(model, val_step_func=_val_step_func):
                # patch_merge in model
                sr = model(**data, cfg=cfg, mode="sharpening_eval", patch_merge=True)
            else:
                raise NotImplementedError(
                    "model should have 'forward_chop' or 'patch_merge' arg in @val_step"
                )
        # normal inference
        else:
            if not lifting_full_resolution:
                from utils.unsupervised_sharpening_loss import (
                    unsupervised_finetune_inner_loop,
                )

                keep_model_across_sample = lifting_full_resolution_kwargs.pop(
                    "keep_model_across_sample", False
                )

                first_hrms_est = None
                lft_unsup_typ = lifting_full_resolution_kwargs.pop(
                    "unsupervised_type", "projection"
                )
                discard_first_est_hrms = lifting_full_resolution_kwargs.get(
                    "discard_first_est_hrms", False
                )
                n_iter = lifting_full_resolution_kwargs.pop("n_iter", 5)
                if lft_unsup_typ == "hrms" or not discard_first_est_hrms:
                    first_hrms_est = model(
                        **data, cfg=cfg, mode="sharpening_eval", patch_merge=False
                    )

                ## inner unsupervised pansharpening loop
                final_hrms, ft_model = unsupervised_finetune_inner_loop(
                    data["pan"],
                    data["ms"],
                    data["lms"],
                    hrms=first_hrms_est,
                    n_iter=n_iter,
                    finetune_model=model,
                    unsupervised_type=lft_unsup_typ,
                    sensor=sensor,
                    discard_first_est_hrms=discard_first_est_hrms,  # force to be True
                    # other kwargs
                    loss_fn_kwargs=lifting_full_resolution_kwargs,
                    other_model_inputs=data,
                )
                if keep_model_across_sample:
                    model = ft_model
                sr = final_hrms
            else:
                # not split patch nor lifting full resolution
                sr = model(**data, cfg=cfg, mode="sharpening_eval", patch_merge=False)

        # post-process
        sr = sr.clip(0, 1)
        sr1 = sr.detach().cpu().numpy()
        all_sr.append(sr1)

        # feature callbacks (e.g., saving features)
        if feature_callback is not None:
            feature_callback(model, i)

        # metrics
        analysis(sr, data["ms"], data["lms"], data["pan"])

    logger.info(analysis.print_str())

    return all_sr


@torch.no_grad()
def ref_for_loop(
    model: "BaseModel",
    dl: DataLoader,
    cfg: "NameSpace | None" = None,
    *,
    device: "str | torch.device | None" = None,
    split_patch=False,
    ergas_ratio=4,
    feature_callback=None,
    patch_merge_module_kwargs: dict = {},
):
    from model.base_model import PatchMergeModule

    analysis = AnalysisPanAcc(ergas_ratio)
    all_sr = []
    inference_bar = tqdm(enumerate(dl, 1), dynamic_ncols=True, total=len(dl))
    if device is None:
        device = next(iter(model.parameters())).device

    _val_step_func = patch_merge_module_kwargs.pop("val_step_func", "val_step")
    if not (
        has_patch_merge_model(model)
        or patch_merge_in_val_step(model, val_step_func=_val_step_func)
    ):
        # assert bs == 1, 'batch size should be 1'
        # warp the model into PatchMergeModule
        model = PatchMergeModule(net=model, device=device, **patch_merge_module_kwargs)
    for i, data in inference_bar:
        data = dict_data_to_device_and_type(data, device, dtype=torch.float32)
        gt = data["gt"]

        # split the image into several patches to avoid gpu OOM
        if split_patch:
            if hasattr(model, "forward_chop") or isinstance(model, PatchMergeModule):
                input = (data["ms"], data["lms"], data["pan"])
                sr = model.forward_chop(*input)[0]
            elif patch_merge_in_val_step(model, val_step_func=_val_step_func):
                sr = model(**data, cfg=cfg, mode="sharpening_eval", patch_merge=True)
            else:
                raise NotImplementedError(
                    "model should have @forward_chop or patch_merge arg in @val_step"
                )
        # no split patch
        else:
            if patch_merge_in_val_step(model, val_step_func=_val_step_func):
                # has patch merge model in model registered, but we do not use it
                sr = model(**data, cfg=cfg, mode="sharpening_eval", patch_merge=False)
            else:
                sr = model(**data, cfg=cfg, mode="sharpening_eval", patch_merge=False)

        if feature_callback is not None:
            feature_callback(model, i)

        sr = sr.clip(0, 1)
        sr1 = sr.detach().cpu().numpy()
        all_sr.append(sr1)

        analysis(gt, sr)

    print(analysis.print_str())

    return all_sr


def main(args):
    accelerator = Accelerator(
        dataloader_config=DataLoaderConfiguration(
            split_batches=False, even_batches=False, non_blocking=True
        )
    )
    args.device = str(accelerator.device)

    dataset_type = args.dataset
    assert dataset_type in [
        "wv3",
        "gf2",
        "qb",
        "cave",
        "harvard",
        "cave_x8",
        "harvard_x8",
        "gf5",
        "chikusei",
        "pavia",
        "botswana",
    ]
    if_hisi = dataset_type in ["cave", "cave_x8", "harvard", "harvard_x8", "gf5"]

    # configs
    save_format = args.save_format
    full_res = args.full_res
    split_patch = args.split_patch
    patch_size = args.patch_size
    ergas_ratio = 4
    save_mat = args.save_mat

    # unused if patch_merge_module in model
    _patch_size_list = [
        patch_size // ergas_ratio,
        patch_size // 2,
        patch_size,
        patch_size,
    ]  # ms, lms, pan

    dl_bs = args.val_bs
    load_from_logs = args.load_from_logs
    crop_bs = args.crop_bs

    ## print configurations
    logger.info("[green]Inference Configurations:[/green]")
    logger.info(args)
    print("=" * 90)

    ## get dataset
    ds, _ = get_eval_dataset(args)
    dl = DataLoader(ds, batch_size=dl_bs, shuffle=False, num_workers=0)

    ## inference loop function
    loop_func = (
        partial(
            ref_for_loop,
            split_patch=split_patch,
            ergas_ratio=ergas_ratio,
            device=accelerator.device,
            cfg=args,
        )
        if not full_res  # if is using full resolution, we use unref-for-loop
        else partial(
            unref_for_loop,
            sensor=dataset_type,
            device=accelerator.device,
            cfg=args,
        )
    )

    # * old model loading ==========================
    # new model loading
    if args.model_class is None:
        model = build_network(args.full_arch, **args.network_configs)
    else:
        logger.info(
            f"loading model from class: {args.model_class} with config file: {args.config_file}"
        )
        model_cls = hydra.utils.get_class(args.network_configs.model_cls_module)
        model = model_cls(**OmegaConf.to_container(args.model_init_kwargs))

    # -------------------load params-----------------------
    # params = torch.load(p, map_location=device)
    # odict = OrderedDict()
    # for k, v in params['model'].items():
    #    odict['module.' + k] = v
    # model.load_state_dict(params["model"])

    model = model.to(args.device)
    model = module_load(args.model_path, model, args.device, strict=True)
    model.eval()
    # -----------------------------------------------------

    # -------------------inference-------------------------
    model, dl = accelerator.prepare(model, dl, device_placement=[True, True])
    all_sr = loop_func(model, dl, split_patch=split_patch)
    # -----------------------------------------------------

    # -------------------save result-----------------------
    d = {}

    # max value of each sensor
    if dataset_type in ["wv3", "qb", "wv2"]:
        const = 2047.0
    elif dataset_type in ["gf2"]:
        const = 1023.0
    else:
        # for HMIF, the max values is 1.0
        const = 1.0

    cat_sr = np.concatenate(all_sr, axis=0).astype("float32")
    d["sr"] = np.asarray(cat_sr) * const
    try:
        d["gt"] = np.asarray(ds.gt[:]) * const
    except AttributeError:
        print("no gt")

    if save_mat:
        _ref_or_not_s = "unref" if full_res else "ref"
        _patch_size_s = f"_p{patch_size}" if split_patch else ""
        if dataset_type not in [
            "cave_x4",
            "harvard_x4",
            "cave_x8",
            "harvard_x8",
            "bostwana",
            "pavia",
            "chikusei",
            "botswana",
            "gf5",
        ]:
            d["ms"] = np.asarray(ds.ms[:]) * const
            d["lms"] = np.asarray(ds.lms[:]) * const
            d["pan"] = np.asarray(ds.pan[:]) * const
        else:
            d["ms"] = np.asarray(ds.lr_hsi[:]) * const
            d["lms"] = np.asarray(ds.hsi_up[:]) * const
            d["pan"] = np.asarray(ds.rgb[:]) * const

        if save_format == "mat":
            path = f"./visualized_img/{args.full_arch}/data_{args.full_arch}_{dataset_type}_{_ref_or_not_s}{_patch_size_s}.mat"
            os.makedirs(os.path.dirname(path), exist_ok=True)

            savemat(path, d)
        else:
            path = f"./visualized_img/{args.full_arch}/data_{args.full_arch}_{dataset_type}_{_ref_or_not_s}{_patch_size_s}.h5"
            save_file = h5py.File(path, "w")
            save_file.create_dataset("sr", data=d["sr"])
            save_file.create_dataset("ms", data=d["ms"])
            save_file.create_dataset("lms", data=d["lms"])
            save_file.create_dataset("pan", data=d["pan"])
            save_file.close()
        print(f"save results in {path}")
    # -----------------------------------------------------


if __name__ == "__main__":
    from utils import catch_any_error

    args = get_args()

    with catch_any_error():
        main(args)
