import zipfile
from collections import OrderedDict
from contextlib import contextmanager
from io import BytesIO
from pathlib import Path
from typing import Union

import h5py
import numpy as np
import PIL.Image as Image
import torch
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from torch.utils.data import ConcatDataset, DataLoader, Dataset

from .cfg_utils import omegaconf_create
from .log_utils import (
    catch_any_error,
    easy_logger,
    generate_id,
)
from .misc import (
    NameSpace,
    default_getattr,
    default_none_getattr,
    dict_to_namespace,
    h5py_to_dict,
    merge_args,
)

FUSION_TASK_DATASETS_MAPPING = {
    "roadscene": "VIF",
    "tno": "VIF",
    "msrs": "VIF",
    "llvip": "VIF",
    "m3fd": "VIF",
    "med_harvard": "medical_fusion",
    "sice": "MEF",
    "mefb": "MEF",
    "realmff": "MFF",
    "mff_whu": "MFF",
    "lytro": "MFF",
    "mff_mffw": "MFF",
    "wv3": "Pansharpening",
    "qb": "Pansharpening",
    "gf2": "Pansharpening",
    "hisi-houston": "Pansharpening",
    "cave_x4": "HMIF",
    "harvard_x4": "HMIF",
    "cave_x8": "HMIF",
    "harvard_x8": "HMIF",
    "pavia": "HMIF",
    "chikusei": "HMIF",
    "botswana": "HMIF",
    # ========= Degradation fusion task ==========
    "dif": "DIF",
    # ======== Eval fusion task =======
    "eval_fusion": "READY_TO_SPECIFY",
}

ALL_TASKS = [
    "Pansharpening",
    "HMIF",
    "VIF",
    "MEF",
    "MFF",
    "medical_fusion",
    "DIF",
]

# *==============================================================
# * train and eval datasets
# *==============================================================


def get_eval_dataset(args: NameSpace, logger=None):
    from task_datasets import (
        GF2Datasets,
        HISRDatasets,
        LLVIPDALIPipeLoader,
        LytroDataset,
        M3FDDALIPipeLoader,
        MedHarvardDataset,
        MEFBDataset,
        MFFWDataset,
        MFFWHUDataset,
        MSRSDatasets,
        RealMFFDataset,
        RoadSceneDataset,
        SICEDataset,
        TNODataset,
        WV3Datasets,
    )

    logger = easy_logger(func_name="get_eval_dataset")

    val_ds, val_dl = None, None

    logger.info(f"use dataset: {args.dataset} on fusion task")
    _prefix_task = FUSION_TASK_DATASETS_MAPPING.get(args.dataset, "UNSUPPORTED")
    if _prefix_task != "READY_TO_SPECIFY":  # spefify by the args.dataset
        args.fusion_task = _prefix_task
    if args.fusion_task == "UNSUPPORTED":
        raise ValueError(f"not support dataset {args.dataset}")
    logger.info(f"dataset {args.dataset} on fusion task {args.fusion_task}")

    # 1. vis-ir image fusion (without gt)
    if args.dataset == "roadscene":
        val_ds = RoadSceneDataset(
            args.path.base_dir,
            args.dataset_mode,
            no_split=True,
            get_name=True,
            only_resize=None,
            output_none_mask=True,
            with_txt_feature=default_getattr(
                args, "datasets_cfg.roadscene.with_txt_feature", False
            ),
        )
    elif args.dataset == "tno":
        val_ds = TNODataset(
            args.path.base_dir,
            args.dataset_mode,
            aug_prob=0.0,
            no_split=True,
            get_name=True,
            output_none_mask=False,
            with_txt_feature=default_getattr(
                args, "datasets_cfg.tno.with_txt_feature", False
            ),
            only_resize=None,
        )
    elif args.dataset == "msrs":
        val_ds = MSRSDatasets(
            args.path.base_dir,
            mode=args.dataset_mode,  # or 'test'/'detection'
            transform_ratio=0.0,
            get_name=True,
            reduce_label=default_getattr(args, "datasets_cfg.msrs.reduce_label", True),
            with_txt_feature=default_getattr(
                args, "datasets_cfg.msrs.with_txt_feature", False
            ),
            with_mask=default_getattr(args, "datasets_cfg.msrs.with_mask", False),
        )
    elif args.dataset == "llvip":
        args.fusion_task = "VIF"
        val_dl = LLVIPDALIPipeLoader(
            args.path.base_dir,
            args.dataset_mode,
            batch_size=args.val_bs,
            device=args.device,
            shuffle=False,
            with_mask=default_getattr(args, "datasets_cfg.llvip.with_mask", False),
            reduce_label=default_getattr(args, "datasets_cfg.llvip.reduce_label", True),
            with_txt_feature=default_getattr(
                args, "datasets_cfg.llvip.with_txt_feature", False
            ),
            get_name=True,
        )
    elif args.dataset == "m3fd":
        val_dl = M3FDDALIPipeLoader(
            args.path.base_dir,
            args.dataset_mode,
            batch_size=args.val_bs,
            device=args.device,
            shuffle=False,
            with_mask=default_getattr(args, "datasets_cfg.m3fd.with_mask", False),
            reduce_label=default_getattr(args, "datasets_cfg.m3fd.reduce_label", True),
            with_txt_feature=default_getattr(
                args, "datasets_cfg.m3fd.with_txt_feature", False
            ),
            get_name=True,
        )

    elif args.dataset == "med_harvard":
        val_ds = MedHarvardDataset(
            args.path.base_dir,
            mode="test",
            device=args.device,
            data_source="xmu",
            get_name=True,
            task=default_getattr(args, "datasets_cfg.med_harvard.task", None),
            with_mask=default_getattr(
                args, "datasets_cfg.med_harvard.with_mask", False
            ),
            reduce_label=default_getattr(
                args, "datasets_cfg.med_harvard.reduce_label", True
            ),
            with_txt_feature=default_getattr(
                args, "datasets_cfg.med_harvard.with_txt_feature", False
            ),
            only_resize=default_getattr(
                args, "datasets_cfg.med_harvard.only_resize", None
            ),
        )
    elif args.dataset == "sice":
        val_ds = SICEDataset(
            data_dir=args.path.base_dir,
            mode="test",
            transform_ratio=0.0,
            with_mask=default_getattr(args, "datasets_cfg.sice.with_mask", False),
            with_txt=default_getattr(args, "datasets_cfg.sice.with_txt", False),
            only_y=default_getattr(args, "datasets_cfg.sice.only_y", False),
            get_name=True,
        )
    elif args.dataset == "mefb":
        val_ds = MEFBDataset(
            data_dir=args.path.base_dir,
            mode=args.dataset_mode,
            transform_ratio=0.0,
            with_mask=default_getattr(args, "datasets_cfg.mefb.with_mask", False),
            with_txt=default_getattr(args, "datasets_cfg.mefb.with_txt", False),
            only_y=default_getattr(args, "datasets_cfg.mefb.only_y", False),
            get_name=True,
        )
    elif args.dataset == "realmff":
        val_ds = RealMFFDataset(
            data_dir=args.path.base_dir,
            mode=args.dataset_mode,
            transform_ratio=0.0,
            with_mask=default_getattr(args, "datasets_cfg.realmff.with_mask", False),
            with_txt=default_getattr(args, "datasets_cfg.realmff.with_txt", False),
            get_name=True,
        )
    elif args.dataset == "mff_whu":
        val_ds = MFFWHUDataset(
            data_dir=args.path.base_dir,
            mode=args.dataset_mode,
            transform_ratio=0.0,
            with_mask=default_getattr(args, "datasets_cfg.mff_whu.with_mask", False),
            with_txt=default_getattr(args, "datasets_cfg.mff_whu.with_txt", False),
            get_name=True,
        )
    elif args.dataset == "lytro":
        val_ds = LytroDataset(
            data_dir=args.path.base_dir,
            with_mask=default_getattr(args, "datasets_cfg.lytro.with_mask", False),
            with_txt=default_getattr(args, "datasets_cfg.lytro.with_txt", False),
            get_name=True,
        )
    elif args.dataset == "mff_mffw":
        val_ds = MFFWDataset(
            data_dir=args.path.base_dir,
            with_mask=default_getattr(args, "datasets_cfg.mff_mffw.with_mask", False),
            with_txt=default_getattr(args, "datasets_cfg.mff_mffw.with_txt", False),
            get_name=True,
        )
    elif args.dataset == "eval_fusion":
        from task_datasets.unify_fusion_datasets import FusionDirDataset

        val_ds = FusionDirDataset(
            modality_dirs=args.modality_dirs,
        )

    ## 2. sharpening datasets (with gt)
    elif args.dataset in [
        "wv3",
        "qb",
        "gf2",
        "cave_x4",
        "harvard_x4",
        "cave_x8",
        "harvard_x8",
        "hisi-houston",
        "pavia",
        "chikusei",
        "botswana",
    ]:
        args.fusion_task = "Pansharpening"

        logger.info(f"use dataset: {args.dataset} on pansharpening/HISR task")
        # FIXME: 需要兼顾老代码（只有trian_path和val_path）的情况
        if hasattr(args.path, "val_path"):
            # 旧代码：手动切换数据集路径
            val_path = args.path.val_path
        else:
            if not args.full_res:
                val_path = default_none_getattr(args, f"path.{args.dataset}_val_path")
            else:
                val_path = default_none_getattr(args, f"path.{args.dataset}_full_path")

        assert val_path is not None, "val_path should not be None"

        if val_path is not None:
            assert val_path.endswith(".h5"), 'val_path should end with ".h5"'

        h5_val = h5py.File(val_path)

        # 1. parsharpening
        if args.dataset in ["wv3", "qb"]:
            d_val = h5py_to_dict(h5_val)
            val_ds = WV3Datasets(
                d_val,
                hp=default_getattr(args, f"datasets_cfg.{args.dataset}.hp", False),
                full_res=default_getattr(args, "full_res", False),
                aug_prob=0.0,
                txt_file=default_none_getattr(
                    args, f"path.{args.dataset}_txt_full_path"
                ),
            )
        elif args.dataset == "gf2":
            d_val = h5py_to_dict(h5_val)
            val_ds = GF2Datasets(
                d_val,
                hp=default_getattr(args, "datasets_cfg.gf2.hp", False),
                full_res=default_getattr(args, "full_res", False),
                aug_prob=0.0,
                txt_file=default_none_getattr(args, "path.gf2_txt_val_path"),
                txt_feature_online_load=True,
            )  # txt feature is too large to load into memory

        # 2. hyperspectral image fusion
        elif (
            args.dataset[:4] == "cave"
            or args.dataset[:7] == "harvard"
            or args.dataset[:8] == "chikusei"
            or args.dataset[:5] == "pavia"
            or args.dataset[:8] == "botswana"
            or args.dataset[:7] == "houston"
        ):
            args.fusion_task = "HMIF"

            if args.dataset in ["pavia", "botswana"]:
                keys = ["GT", "MS", "PAN", "LMS"]
            else:
                keys = ["LRHSI", "HSI_up", "RGB", "GT"]

            if args.dataset.split("-")[-1] == "houston":
                from einops import rearrange

                # to avoid unpicklable error for clourse in class
                def permute_fn(x):
                    return rearrange(x, "b h w c -> b c h w")

                dataset_fn = permute_fn
            else:
                dataset_fn = None

            d_val = h5py_to_dict(h5_val, keys)
            dataset_txt_paths = {
                "cave": "path.cave_txt_val_path",
                "harvard": "path.harvard_txt_val_path",
                "chikusei": "path.chikusei_txt_val_path",
                "pavia": "path.pavia_txt_val_path",
                "botswana": "path.botswana_txt_val_path",
                "houston": "path.houston_txt_val_path",
            }

            txt_file_path = dataset_txt_paths.get(args.dataset, None)
            txt_file = default_none_getattr(args, txt_file_path)

            val_ds = HISRDatasets(
                d_val,
                txt_file=txt_file,
                dataset_fn=dataset_fn,
                dataset_name=args.dataset,
            )

    else:
        raise NotImplementedError(f"not support dataset {args.dataset}")

    return val_ds, val_dl


def get_fusion_dataset(args: "NameSpace | DictConfig"):
    from accelerate import PartialState

    state = PartialState()
    device = state.device

    train_ds, val_ds, train_dl, val_dl = None, None, None, None

    if args.dataset in [
        "flir",
        "tno",
        "roadscene_tno_joint",
        "vis_ir_joint",
        "msrs",
        "llvip",
        "med_harvard",
        "m3fd",
        "sice",
        "mefb",
        "realmff",
        "mff_whu",
    ]:
        args.task = "fusion"
        args.path.base_dir = getattr(args.path, f"{args.dataset}_base_dir")

        if args.dataset == "roadscene":
            args.fusion_task = "VIF"
            from task_datasets import RoadSceneDataset

            train_ds = RoadSceneDataset(args.path.base_dir, "train")
            val_ds = RoadSceneDataset(args.path.base_dir, "test")

        elif args.dataset in ["tno", "roadscene_tno_joint"]:
            from task_datasets import TNODataset

            args.fusion_task = "VIF"
            train_ds = TNODataset(
                args.path.base_dir,
                "train",
                aug_prob=args.aug_probs[0],
                duplicate_vis_channel=True,
            )
            val_ds = Dataset(
                args.path.base_dir,
                "test",
                aug_prob=args.aug_probs[1],
                no_split=True,
                duplicate_vis_channel=True,
            )

        elif args.dataset == "msrs":
            from task_datasets.VIF.MSRS import MSRSDatasets

            ds_kwargs = {
                "reduce_label": default_getattr(
                    args, "datasets_cfg.msrs.reduce_label", True
                ),
                "only_resize": default_getattr(
                    args, "datasets_cfg.msrs.only_resize", None
                ),
                "with_txt_feature": default_getattr(
                    args, "datasets_cfg.msrs.with_txt_feature", False
                ),
                "with_mask": default_getattr(
                    args, "datasets_cfg.msrs.with_mask", False
                ),
            }

            args.fusion_task = "VIF"
            train_ds = MSRSDatasets(
                args.path.base_dir,
                "train",
                transform_ratio=default_getattr(
                    args, "datasets_cfg.msrs.transform_ratio", args.aug_probs[0]
                ),
                output_size=default_getattr(
                    args, "datasets_cfg.msrs.output_size", None
                ),
                n_proc_load=1,
                only_y_component=False,
                **ds_kwargs,
            )
            val_ds = MSRSDatasets(
                args.path.base_dir,
                "test",
                transform_ratio=default_getattr(
                    args, "datasets_cfg.msrs.transform_ratio", args.aug_probs[1]
                ),
                output_size=default_getattr(
                    args, "datasets_cfg.msrs.output_size", args.fusion_crop_size
                ),
                fast_eval_n_samples=default_getattr(
                    args, "datasets_cfg.msrs.fast_eval_n_samples", 80
                ),
                n_proc_load=1,
                only_y_component=False,
                **ds_kwargs,
            )

        elif args.dataset == "llvip":
            from task_datasets import LLVIPDALIPipeLoader

            ds_kwargs = {
                "reduce_label": default_getattr(
                    args, "datasets_cfg.llvip.reduce_label", True
                ),
                "only_resize": default_getattr(
                    args, "datasets_cfg.llvip.only_resize", None
                ),
                "with_txt_feature": default_getattr(
                    args, "datasets_cfg.llvip.with_txt_feature", False
                ),
                "with_mask": default_getattr(
                    args, "datasets_cfg.llvip.with_mask", False
                ),
                "crop_strategy": default_getattr(
                    args, "datasets_cfg.llvip.crop_strategy", "crop_resize"
                ),
            }

            args.fusion_task = "VIF"
            # We use DALI pipeline to accelerate the data loading process
            train_dl = LLVIPDALIPipeLoader(
                args.path.base_dir,
                "train",
                batch_size=args.train_bs,
                output_size=args.fusion_crop_size,
                device=state.device,
                num_shards=state.num_processes,
                shard_id=state.process_index,
                shuffle=True,
                only_y_component=False,
                **ds_kwargs,
            )
            val_dl = LLVIPDALIPipeLoader(
                args.path.base_dir,
                "test",
                batch_size=args.val_bs,
                device=state.device,
                fast_eval_n_samples=default_getattr(
                    args, "datasets_cfg.llvip.fast_eval_n_samples", 80
                ),
                num_shards=state.num_processes,
                shard_id=state.process_index,
                shuffle=False,
                only_y_component=False,
                **ds_kwargs,
            )

        elif args.dataset == "m3fd":
            from task_datasets import M3FDDALIPipeLoader

            ds_kwargs = {
                "reduce_label": default_getattr(
                    args, "datasets_cfg.m3fd.reduce_label", True
                ),
                "crop_strategy": default_getattr(
                    args, "datasets_cfg.m3fd.crop_strategy", "crop_resize"
                ),
                "with_txt_feature": default_getattr(
                    args, "datasets_cfg.m3fd.with_txt_feature", False
                ),
                "output_size": default_getattr(
                    args, "datasets_cfg.m3fd.output_size", args.fusion_crop_size
                ),
                "only_resize": default_getattr(
                    args, "datasets_cfg.m3fd.only_resize", None
                ),
                "with_mask": default_getattr(
                    args, "datasets_cfg.m3fd.with_mask", False
                ),
            }

            args.fusion_task = "VIF"
            train_dl = M3FDDALIPipeLoader(
                args.path.base_dir,
                "train",
                batch_size=args.train_bs,
                device=state.device,
                num_shards=state.num_processes,
                shard_id=state.process_index,
                shuffle=True,
                only_y_component=False,
                **ds_kwargs,
            )
            val_dl = M3FDDALIPipeLoader(
                args.path.base_dir,
                "test",
                batch_size=args.val_bs,
                device=state.device,
                num_shards=state.num_processes,
                shard_id=state.process_index,
                shuffle=True,
                only_y_component=False,
                fast_eval_n_samples=default_getattr(
                    args, "datasets_cfg.m3fd.fast_eval_n_samples", 80
                ),
                **ds_kwargs,
            )

        elif args.dataset == "vis_ir_joint":
            from task_datasets import VISIRJointGenericLoader

            additional_args = {
                "reduce_label": default_getattr(
                    args, "datasets_cfg.reduce_label", True
                ),
                "crop_strategy": default_getattr(
                    args, "datasets_cfg.vis_ir_joint.crop_strategy", "crop_resize"
                ),
                "with_txt_feature": default_getattr(
                    args, "datasets_cfg.vis_ir_joint.with_txt_feature", False
                ),
                "output_size": default_getattr(
                    args, "datasets_cfg.vis_ir_joint.output_size", None
                ),
                "only_resize": default_getattr(
                    args, "datasets_cfg.vis_ir_joint.only_resize", None
                ),
                "with_mask": default_getattr(
                    args, "datasets_cfg.vis_ir_joint.with_mask", True
                ),
            }

            args.fusion_task = "VIF"
            train_dl = VISIRJointGenericLoader(
                vars(args.path.base_dir)
                if isinstance(args.path.base_dir, NameSpace)
                else OmegaConf.to_container(args.path.base_dir),
                mode="train",
                batch_size=args.train_bs,
                device=state.device,
                shuffle_in_dataset=True,
                num_shards=state.num_processes,
                shard_id=state.process_index,
                only_y_component=False,
                **additional_args,
            )
            val_dl = VISIRJointGenericLoader(
                ## only test msrs and roadscene_tno_joint dataset
                {
                    "msrs": args.path.base_dir["msrs"],
                    "roadscene_tno_joint": args.path.base_dir["roadscene_tno_joint"],
                },
                mode="test",
                output_size=None,  # enforce the different size images to be the same size or batch size to be 1
                batch_size=args.val_bs,
                device=state.device,
                num_shards=state.num_processes,
                shard_id=state.process_index,
                only_y_component=False,
                crop_strategy=default_getattr(
                    args, "datasets_cfg.vis_ir_joint.crop_strategy", "crop_resize"
                ),
                random_datasets_getitem=default_getattr(
                    args, "datasets_cfg.vis_ir_joint.random_datasets_getitem", False
                ),
                shuffle_in_dataset=True,  # ensure different images in the same batch for tensorboard visualization
                reduce_label=default_getattr(args, "datasets_cfg.reduce_label", True),
                fast_eval_n_samples=default_getattr(
                    args, "datasets_cfg.vis_ir_joint.fast_eval_n_samples", 80
                ),
                with_txt_feature=default_getattr(
                    args, "datasets_cfg.vis_ir_joint.with_txt_feature", False
                ),
                only_resize=default_getattr(
                    args, "datasets_cfg.vis_ir_joint.only_resize", None
                ),
            )

        elif args.dataset == "med_harvard":
            from task_datasets import MedHarvardDataset

            task = default_none_getattr(args, "datasets_cfg.med_harvard.task")
            additional_args = {
                "with_mask": default_getattr(
                    args, "datasets_cfg.med_harvard.with_mask", False
                ),
                "reduce_label": default_getattr(
                    args, "datasets_cfg.med_harvard.reduce_label", True
                ),
                "with_txt_feature": default_getattr(
                    args, "datasets_cfg.med_harvard.with_txt_feature", False
                ),
                "only_resize": default_getattr(
                    args, "datasets_cfg.med_harvard.only_resize", None
                ),
                "device": "cpu",  # pin memory in dataloader
            }

            args.fusion_task = "medical_fusion"  # 'SPECT-MRI', 'PET-MRI', 'CT-MRI'
            train_ds = MedHarvardDataset(
                args.path.base_dir,
                mode="train",
                data_source="xmu",
                transform_ratio=args.aug_probs[0],
                task=task,
                **additional_args,
            )
            val_ds = MedHarvardDataset(
                args.path.base_dir,
                mode="test",
                data_source="xmu",
                transform_ratio=args.aug_probs[1],
                task=task,
                **additional_args,
            )

        elif args.dataset == "sice":
            from task_datasets import SICEDataset

            additional_args = {
                "only_y": default_getattr(args, "datasets_cfg.sice.only_y", False),
                "use_gt": default_getattr(args, "datasets_cfg.sice.use_gt", False),
                "with_txt": default_getattr(args, "datasets_cfg.sice.with_txt", False),
                "with_mask": default_getattr(
                    args, "datasets_cfg.sice.with_mask", False
                ),
                "stop_aug_when_n_iters": default_getattr(
                    args, "datasets_cfg.sice.stop_aug_when_n_iters", -1
                ),
            }

            args.fusion_task = "MEF"
            train_ds = SICEDataset(
                data_dir=args.path.base_dir,
                mode="all",
                transform_ratio=args.aug_probs[0],
                output_size=default_getattr(args, "datasets_cfg.sice.output_size", 128),
                **additional_args,
            )
            val_ds = SICEDataset(
                data_dir=args.path.base_dir,
                mode="test",
                transform_ratio=args.aug_probs[1],
                **additional_args,
            )

        elif args.dataset == "mefb":
            from task_datasets import MEFBDataset

            additional_args = {
                "only_y": default_getattr(args, "datasets_cfg.mefb.only_y", False),
                "with_mask": default_getattr(
                    args, "datasets_cfg.mefb.with_mask", False
                ),
                "with_txt": default_getattr(args, "datasets_cfg.mefb.with_txt", False),
                "stop_aug_when_n_iters": default_getattr(
                    args, "datasets_cfg.mefb.stop_aug_when_n_iters", -1
                ),
            }

            args.fusion_task = "MEF"
            train_ds = MEFBDataset(
                data_dir=args.path.base_dir,
                mode="all",
                transform_ratio=args.aug_probs[0],
                output_size=default_getattr(args, "datasets_cfg.mefb.output_size", 128),
                **additional_args,
            )
            val_ds = MEFBDataset(
                data_dir=args.path.base_dir,
                mode="test",
                transform_ratio=args.aug_probs[1],
                **additional_args,
            )

        elif args.dataset == "realmff":
            from task_datasets import RealMFFDataset

            additional_args = {
                "with_mask": default_getattr(
                    args, "datasets_cfg.realmff.with_mask", False
                ),
                "with_txt": default_getattr(
                    args, "datasets_cfg.realmff.with_txt", False
                ),
                "use_gt": default_getattr(args, "datasets_cfg.realmff.use_gt", False),
                "stop_aug_when_n_iters": default_getattr(
                    args, "datasets_cfg.realmff.stop_aug_when_n_iters", -1
                ),
            }

            args.fusion_task = "MFF"
            train_ds = RealMFFDataset(
                data_dir=args.path.base_dir,
                mode="all",
                transform_ratio=args.aug_probs[0],
                output_size=default_getattr(
                    args, "datasets_cfg.realmff.output_size", 128
                ),
                **additional_args,
            )
            val_ds = RealMFFDataset(
                data_dir=args.path.base_dir,
                mode="test",
                transform_ratio=args.aug_probs[1],
                **additional_args,
            )

        elif args.dataset == "mff_whu":
            from task_datasets import MFFWHUDataset

            additional_args = {
                "use_gt": default_getattr(args, "datasets_cfg.mff_whu.use_gt", False),
                "with_mask": default_getattr(
                    args, "datasets_cfg.mff_whu.with_mask", False
                ),
                "with_txt": default_getattr(
                    args, "datasets_cfg.mff_whu.with_txt", False
                ),
            }

            args.fusion_task = "MFF"
            train_ds = MFFWHUDataset(
                data_dir=args.path.base_dir,
                mode="all",
                transform_ratio=args.aug_probs[0],
                output_size=default_getattr(
                    args, "datasets_cfg.mff_whu.output_size", 128
                ),
                **additional_args,
            )
            val_ds = MFFWHUDataset(
                data_dir=args.path.base_dir,
                mode="test",
                transform_ratio=args.aug_probs[1],
                **additional_args,
            )

        elif args.dataset == "unify_fusion":
            from task_datasets.unify_fusion_datasets import make_unify_dataloader

            args.fusion_task = None
            args.task = "UnifyFusion"

            datasets_kwargs = {
                "root_of_dirs": default_getattr(
                    args, "datasets_cfg.unify_fusion_datasets.root_of_dirs", None
                ),
                "aug_prob": default_getattr(
                    args, "datasets_cfg.unify_fusion_datasets.aug_prob", 0.3
                ),
                # NOTE: hack `transforms`, `augmentations_pipes`, and `is_valid_file` mannualy in your main script
                "transforms": default_getattr(
                    args, "datasets_cfg.unify_fusion_datasets.transforms", None
                ),
                "augmentations_pipes": default_getattr(
                    args, "datasets_cfg.unify_fusion_datasets.augmentations_pipes", None
                ),
                "is_valid_file": default_getattr(
                    args, "datasets_cfg.unify_fusion_datasets.is_valid_file", None
                ),
            }
            dataloader_kwargs = {
                "batch_size": args.train_bs,
                "num_workers": args.num_workers,
                "pin_memory": True,
                "shuffle": args.shuffle if not state.use_distributed else None,
                "drop_last": True if args.shuffle else False,
            }
            train_dl = make_unify_dataloader(
                root_of_dirs=args.path.base_dir,
                datasets_kwargs=datasets_kwargs,
                dataloader_kwargs=dataloader_kwargs,
            )
            val_ds = None
            args._construct_dataloader = False
        else:
            raise NotImplementedError(f"not support dataset {args.dataset}")

    elif args.dataset in [
        "wv3",
        "qb",
        "gf2",
        "cave_x4",
        "harvard_x4",
        "cave_x8",
        "harvard_x8",
        "hisi-houston",
        "pavia",
        "chikusei",
        "botswana",
    ]:
        args.task = "sharpening"

        # the dataset has already splitted
        # FIXME: 需要兼顾老代码（只有trian_path和val_path）的情况
        if hasattr(args.path, "train_path") and hasattr(args.path, "val_path"):
            # 旧代码：手动切换数据集路径
            train_path = args.path.train_path
            val_path = args.path.val_path
        else:
            _args_path_keys = list(args.path.__dict__.keys())
            for k in _args_path_keys:
                if args.dataset in k:
                    train_path = getattr(args.path, f"{args.dataset}_train_path")
                    val_path = getattr(args.path, f"{args.dataset}_val_path")
        assert (
            train_path is not None and val_path is not None
        ), "train_path and val_path should not be None"

        h5_train, h5_val = (
            h5py.File(train_path),
            h5py.File(val_path),
        )

        if args.dataset in ["wv3", "qb"]:
            from task_datasets import WV3Datasets

            args.fusion_task = "Pansharpening"

            txt_feature_online_load = True if args.dataset == "qb" else False
            d_train, d_val = h5py_to_dict(h5_train), h5py_to_dict(h5_val)
            train_ds, val_ds = (
                WV3Datasets(
                    d_train,
                    aug_prob=args.aug_probs[0],
                    txt_file=default_none_getattr(
                        args, f"path.{args.dataset}_txt_train_path"
                    ),
                    txt_feature_online_load=txt_feature_online_load,
                ),
                WV3Datasets(
                    d_val,
                    aug_prob=args.aug_probs[1],
                    txt_file=default_none_getattr(
                        args, f"path.{args.dataset}_txt_val_path"
                    ),
                    txt_feature_online_load=txt_feature_online_load,
                ),
            )

        elif args.dataset == "gf2":
            from task_datasets import GF2Datasets

            args.fusion_task = "Pansharpening"
            d_train, d_val = h5py_to_dict(h5_train), h5py_to_dict(h5_val)
            train_ds, val_ds = (
                GF2Datasets(
                    d_train,
                    aug_prob=args.aug_probs[0],
                    txt_file=default_none_getattr(args, "path.gf2_txt_train_path"),
                    txt_feature_online_load=True,
                ),
                GF2Datasets(
                    d_val,
                    aug_prob=args.aug_probs[1],
                    txt_file=default_none_getattr(args, "path.gf2_txt_val_path"),
                    txt_feature_online_load=True,
                ),
            )

        elif (
            args.dataset[:4] == "cave"
            or args.dataset[:7] == "harvard"
            or args.dataset[:8] == "chikusei"
            or args.dataset[:5] == "pavia"
            or args.dataset[:8] == "botswana"
            or args.dataset[:7] == "houston"
        ):
            from task_datasets import HISRDatasets

            args.fusion_task = "HMIF"

            if args.dataset.split("-")[-1] == "houston":
                from einops import rearrange

                def permute_fn(x):
                    return rearrange(x, "b h w c -> b c h w")

                dataset_fn = permute_fn
            else:
                dataset_fn = None

            ## get txt file path
            def get_dataset_txt_paths(mode, dataset_name):
                return {
                    "cave": f"path.cave_txt_{mode}_path",
                    "harvard": f"path.harvard_txt_{mode}_path",
                    "chikusei": f"path.chikusei_txt_{mode}_path",
                    "pavia": f"path.pavia_txt_{mode}_path",
                    "botswana": f"path.botswana_txt_{mode}_path",
                    "houston": f"path.houston_txt_{mode}_path",
                }.get(dataset_name, None)

            ## get dataset dict
            d_train, d_val = (
                h5py_to_dict(h5_train),
                h5py_to_dict(h5_val),
            )

            # large files, close it
            h5_train.close()
            h5_val.close()

            ## get txt path
            txt_file_path_train = get_dataset_txt_paths(
                mode="train", dataset_name=args.dataset
            )
            txt_file_path_val = get_dataset_txt_paths(
                mode="val", dataset_name=args.dataset
            )
            assert (
                txt_file_path_train is not None and txt_file_path_val is not None
            ), "txt_file_path_train and txt_file_path_val should not be None"

            txt_file_train = default_none_getattr(args, txt_file_path_train)
            txt_file_val = default_none_getattr(args, txt_file_path_val)

            ## get datasets
            train_ds = HISRDatasets(
                d_train,
                aug_prob=args.aug_probs[0],
                txt_file=txt_file_train,
                dataset_fn=dataset_fn,
                dataset_name=args.dataset,
            )
            val_ds = HISRDatasets(
                d_val,
                aug_prob=args.aug_probs[1],
                txt_file=txt_file_val,
                dataset_fn=dataset_fn,
                dataset_name=args.dataset,
            )
    elif args.dataset.endswith("vae"):  # 'vq_vae' or 'lfq_vae'
        from torch.utils.data import random_split

        from task_datasets.unify_fusion_datasets import make_unify_dataloader

        unify_dataset_kwargs = {
            "root_of_dirs": default_getattr(
                args, "datasets_cfg.vae.root_of_dirs", None
            ),
            "aug_prob": default_getattr(args, "datasets_cfg.vae.aug_prob", 0.0),
        }
        unify_dataloader_kwargs = {
            "batch_size": args.train_bs,
            "num_workers": args.num_workers,
            "pin_memory": True,
            "shuffle": args.shuffle,  # if not state.use_distributed else None,
            "drop_last": True,  # if args.shuffle else False,
        }
        train_dl, _ = make_unify_dataloader(
            unify_dataset_kwargs=unify_dataset_kwargs,
            dataloader_kwargs=unify_dataloader_kwargs,
        )
        train_ds = train_dl.dataset
        # split as validation dataset but still in training dataloader
        val_ds = random_split(train_ds, [0.05, 0.95])[0]
        val_dl = DataLoader(val_ds, **unify_dataloader_kwargs)
        args._construct_dataloader = False

    elif args.dataset.upper().startswith("DIF"):  # 'dif_*' dataset
        from task_datasets.DIF.degraded_wds import get_degraded_wds_loader

        train_cfg = args.dataset_cfg.train
        val_cfg = args.dataset_cfg.val
        train_ds, train_dl = get_degraded_wds_loader(**train_cfg)
        val_ds, val_dl = get_degraded_wds_loader(**val_cfg)
        args._construct_dataloader = False

    else:
        raise NotImplementedError(f"not support dataset {args.dataset}")

    ## Dataloader
    if default_getattr(args, "_construct_dataloader", True):
        train_sampler, val_sampler = None, None
        seed = args.default_getattr("seed", None)
        loader_generator = (
            torch.Generator().manual_seed(seed) if seed is not None else None
        )
        n_worker = default_getattr(args, "num_workers", 0)
        if train_dl is None and val_ds is not None:
            train_dl = DataLoader(
                train_ds,
                args.train_bs,
                num_workers=n_worker,
                sampler=train_sampler,
                pin_memory=True,
                shuffle=args.shuffle,  # if not state.use_distributed else None,
                drop_last=True if args.shuffle else False,
                prefetch_factor=8 if n_worker > 0 else None,
                persistent_workers=True if n_worker > 0 else False,
                generator=loader_generator,
            )
        if val_dl is None and val_ds is not None:
            val_dl = DataLoader(
                val_ds,
                args.val_bs,  # assert bs is 1, when using PatchMergeModule
                num_workers=0,
                sampler=val_sampler,
                pin_memory=False,
                shuffle=args.shuffle,  # if not state.use_distributed else None,
                drop_last=False,
                generator=loader_generator,
            )

    if "_construct_dataloader" in args:
        delattr(args, "_construct_dataloader")

    return train_ds, train_dl, val_ds, val_dl


# *==============================================================
# * dataset utilities
# *==============================================================


class ChainDataset(Dataset):
    def __init__(self, datasets: list[Dataset]) -> None:
        super().__init__()
        self.datasets = datasets
        self.prod_len = np.cumsum([len(d) for d in datasets])
        self.prod_len = np.insert(self.prod_len, 0, 0)
        # self.sample_idx_to_dataset_idx = np.digitize(np.arange(self.prod_len[-1]), self.prod_len)
        self.sample_idx_to_dataset_idx = (
            lambda x: np.searchsorted(self.prod_len, x, side="right") - 1
        )

    def __getitem__(self, index):
        dataset_idx = self.sample_idx_to_dataset_idx(index)
        return self.datasets[dataset_idx][index - self.prod_len[dataset_idx]]

    def __len__(self):
        return self.prod_len[-1]


def concat_dataset(datasets: list[Dataset]):
    """concatenate datasets"""
    assert len(datasets) > 0, "datasets should not be empty"

    return ChainDataset(datasets)


def get_train_dataset(
    main_args: "NameSpace | DictConfig",
    init_with_default_ds_cfg: bool = True,
    default_ds_yaml_file: str = "configs/datasets/datasets.yaml",
):
    """get train dataset

    args modified:
        _construct_dataloader: bool (delete after used)
        dataset: str (change if using multi-dataset)

    """
    from accelerate import PartialState

    # accelerate partial state
    state = PartialState()

    # initial dataset arguments
    if init_with_default_ds_cfg:
        args = omegaconf_create(default_ds_yaml_file)
        args = dict_to_namespace(OmegaConf.to_container(args))
        if isinstance(main_args, DictConfig):
            main_args = dict_to_namespace(OmegaConf.to_container(main_args))
        args = merge_args(args, main_args)
    else:
        args = main_args

    # add multi-dataset support
    if "+" in args.dataset:
        _ds_name = args.dataset
        dataset_names = args.dataset.split("+")
        train_datasets = []
        val_datasets = []
        args._construct_dataloader = False
        for dataset_name in dataset_names:
            assert dataset_name in [
                "sice",
                "mefb",
                "realmff",
                "mff_whu",
            ], f"not support dataset {dataset_name}"
            args.dataset = dataset_name
            train_ds, _, val_ds, _ = get_fusion_dataset(args)
            train_datasets.append(train_ds)
            val_datasets.append(val_ds)
        train_ds = concat_dataset(train_datasets)
        val_ds = concat_dataset(val_datasets)

        # dataloader
        n_worker = default_getattr(args, "num_workers", 0)
        train_sampler, val_sampler = None, None  # handled by accelerator when using ddp
        train_dl = DataLoader(
            train_ds,
            args.train_bs,
            num_workers=n_worker,
            sampler=train_sampler,
            pin_memory=True,  # TODO: if pin memory in dataset ready with cuda stream, it will raise error
            shuffle=True,  # args.shuffle if state.use_distributed else None,
            drop_last=True if args.shuffle else False,
            prefetch_factor=8 if n_worker > 0 else None,
            persistent_workers=True if n_worker > 0 else False,
        )
        val_dl = DataLoader(
            val_ds,
            args.val_bs,  # assert bs is 1, when using PatchMergeModule
            num_workers=0,
            sampler=val_sampler,
            pin_memory=False,
            shuffle=True,  # args.shuffle if not state.use_distributed else None,
            drop_last=False,
        )
        args.dataset = _ds_name  # unchange dataset name for logging

        return (train_ds, train_dl, val_ds, val_dl), args
    else:
        # single dataset or tar-ed webdataset
        return get_fusion_dataset(args), args


# *==============================================================
# * model input checks
# *==============================================================


def check_fusion_mask_inp(tensors: dict[str, torch.Tensor], dtype: torch.dtype):
    from accelerate.state import PartialState

    state = PartialState()
    if "mask" in tensors:
        mask = tensors["mask"]
        if mask is not None:
            # tensor mask and shaped as [bs, *, h, w]
            if isinstance(mask, torch.Tensor) and mask.ndim > 2:
                mask = mask.to(state.device, dtype=dtype)
            else:
                mask = None  # force to be None
        tensors["mask"] = mask

    return tensors


def dict_data_to_device_and_type(
    data: dict, device: "str | torch.device | None", dtype: "torch.dtype | None"
):
    if device is None:
        device = "cpu"
    if dtype is None:
        dtype = torch.float32

    for k, v in data.items():
        if isinstance(v, torch.Tensor):
            data[k] = v.to(device=device, dtype=dtype)
        elif isinstance(v, dict):
            data[k] = dict_data_to_device_and_type(v, device, dtype)
        elif isinstance(v, list):
            data[k] = [dict_data_to_device_and_type(x, device, dtype) for x in v]

    return data


# *==============================================================
# * configuration utilities
# *==============================================================

import argparse

import hydra


def get_main_args():
    """
    args entry of sharpening main script
    """
    parser = argparse.ArgumentParser("RWKVFusion")

    # network
    # NOTE: may decrepted
    parser.add_argument("-a", "--arch", type=str, default="pannet")
    parser.add_argument(
        "--sub_arch", default=None, help="panformer sub-architecture name"
    )

    parser.add_argument(
        "-c", "--config_file", type=str, default=None, help="config file path"
    )
    parser.add_argument("-m", "--model_class", type=str, help="model import path")

    # train config
    parser.add_argument(
        "--pretrain_model_path", type=str, default=None, help="pretrained model path"
    )
    parser.add_argument("--non_load_strict", action="store_false", default=True)
    parser.add_argument("-e", "--num_train_epochs", type=int, default=500)
    parser.add_argument("--val_n_epoch", type=int, default=30)
    parser.add_argument("--warm_up_epochs", type=int, default=10)
    parser.add_argument("-l", "--loss", type=str, default="mse")
    parser.add_argument("--pad_window_base", type=int, default=32)
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    parser.add_argument(
        "--checkpoint_every_n",
        default=None,
        type=int,
        help="checkpointing the running state whether the saving condition is met or not "
        "(see the `check_save_fn` in the training function)",
    )
    parser.add_argument(
        "--ckpt_max_limit",
        type=int,
        default=None,
        help="maximum number of checkpoints to keep",
    )
    parser.add_argument(
        "--mix_precison",
        default="fp32",
        choices=["fp32", "fp16"],
        help="mixed precision training",
    )
    parser.add_argument("--sanity_check", action="store_true", default=False)
    parser.add_argument(
        "--regardless_metrics_save",
        action="store_true",
        default=False,
        help="regardless of the metric and save the model when validation process is done",
    )
    # decrepted
    parser.add_argument("--pretrain", action="store_true", default=False)
    parser.add_argument("--pretrain_id", type=str, default=None)

    # resume training config
    parser.add_argument(
        "--resume_path", default=None, required=False, help="path for resuming state"
    )
    # decrepcted
    parser.add_argument("--resume_lr", type=float, required=False, default=None)
    parser.add_argument("--resume_total_epochs", type=int, required=False, default=None)
    parser.add_argument(
        "--nan_no_raise",
        action="store_true",
        default=False,
        help="not raise error when nan loss",
    )

    # path and load
    parser.add_argument(
        "-p", "--path", type=str, default=None, help="only for unsplitted dataset"
    )
    parser.add_argument("--split_ratio", type=float, default=None)
    parser.add_argument(
        "--load", action="store_true", default=False, help="resume training"
    )
    parser.add_argument("--save_base_path", type=str, default="./weight")
    parser.add_argument(
        "--proj_root",
        type=str,
        default="log_file",
        help="proj_root, do not set it",
        required=False,
    )

    # datasets config
    parser.add_argument("--dataset", type=str, default="wv3")
    parser.add_argument(
        "-b", "--batch_size", type=int, default=1028, help="set train and val bs"
    )
    parser.add_argument("--train_bs", type=int, default=None)
    parser.add_argument("--val_bs", type=int, default=None)
    parser.add_argument(
        "--fusion_crop_size",
        type=int,
        default=72,
        help="image cropped size for fusion task",
    )
    parser.add_argument(
        "--only_y_train",
        action="store_true",
        default=False,
        help="train model only on Y channel, distinguish it from `only_y` argument from dataset configuration",
    )
    parser.add_argument("--shuffle", type=bool, default=True)
    parser.add_argument("--fast_eval_n_samples", type=int, default=128)
    parser.add_argument(
        "--aug_probs",
        nargs="+",
        type=float,
        default=[0.0, 0.0],
        help="augmentation probabilities for train and validation dataset",
    )
    parser.add_argument("-s", "--seed", type=int, default=3407)
    parser.add_argument("-n", "--num_workers", type=int, default=8)
    parser.add_argument("--ergas_ratio", type=int, choices=[2, 4, 8, 16, 20], default=4)

    # logger config
    parser.add_argument("--logger_on", action="store_true", default=False)
    parser.add_argument("--proj_name", type=str, default="panformer_wv3")
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument(
        "--resume_id", type=str, default="None", help="resume training id"
    )
    parser.add_argument("--run_id", type=str, default=generate_id())
    parser.add_argument("--watch_log_freq", type=int, default=10)
    parser.add_argument("--watch_type", type=str, default="None")
    parser.add_argument("--log_metrics", action="store_true", default=True)
    parser.add_argument("--debug", action="store_true", default=False)

    # ddp setting
    parser.add_argument("--world-size", type=int, default=2)
    parser.add_argument(
        "--dist-url", default="env://", help="url used to set up distributed training"
    )
    parser.add_argument("--ddp", action="store_true", default=False)

    # some comments
    parser.add_argument("--comment", type=str, required=False, default="")

    return parser.parse_args()


def get_main_args_hydra(
    config_path: str = "../configs/sharpening_cfgs",
    config_name: str = "sharpening_main",
):
    # hydra init
    hydra.initialize(config_path=config_path)
    args = hydra.compose(config_name=config_name)

    # convert to Namespace
    args = dict_to_namespace(OmegaConf.to_container(args))

    return args


# *==============================================================
# * sanity check
# *==============================================================


def set_ema_model_params_with_keys(
    ema_model_params: "dict[str, list[torch.Tensor] | int | float]",
    keys: "list[str]",
    keys_set: list[str] = ["shadow_params"],
):
    """set ema model parameters with keys

    Args:
        ema_model_params (dict[str, list[torch.Tensor] | int | float]): ema model parameters
        keys (list[str]): keys

    Returns:
        dict: ema model parameters with keys
    """
    logger = easy_logger()

    if not isinstance(keys, list):
        keys = list(keys)

    ema_model_params_with_keys = OrderedDict()
    for k in ema_model_params.keys():
        if k in keys_set and k in ema_model_params:
            logger.info(f"set ema_model {k} params with keys")
            params = ema_model_params[k]
            assert params is not None
            assert len(params) == len(
                keys
            ), "ema_model_params and keys should have the same length"

            _params = OrderedDict()
            for mk, p in zip(keys, params):
                _params[mk] = p

            ema_model_params_with_keys[k] = _params
        elif k not in keys_set and k in ema_model_params:
            ema_model_params_with_keys[k] = ema_model_params[k]

    return ema_model_params_with_keys


def run_once(abled=True):
    def _inner(func):
        def _wrapper(*args, **kwargs):
            nonlocal abled
            if not abled:
                return None
            else:
                outs = func(*args, **kwargs)
                abled = False
                return outs

        return _wrapper

    return _inner


def sanity_check(func: callable):
    @run_once()
    def _inner(*args, **kwargs):
        return func(*args, **kwargs)

    return _inner


# * ==========================================================
# * training utils classes
# * ==========================================================


class StepsCounter:
    def __init__(self, step_names: list[str]):
        # all set to 0
        self.step_names = step_names
        for name in step_names:
            setattr(self, f"n_{name}_steps", 0)

    def __repr__(self):
        return f"StepsCounter({self.state_dict()})"

    def state_dict(self):
        return {
            f"n_{name}_steps": getattr(self, f"n_{name}_steps")
            for name in self.step_names
        }

    def load_state_dict(self, state_dict):
        for name in self.step_names:
            assert name in state_dict, f"Key {name} missing in state_dict"
            setattr(self, f"n_{name}_steps", state_dict[f"n_{name}_steps"])

    def update(self, name: str, update_n: int = 1):
        n_step = self.get(name)
        setattr(self, f"n_{name}_steps", n_step + update_n)

    def get(self, name: str):
        step_name = f"n_{name}_steps"
        assert hasattr(self, step_name), f"Key {step_name} missing in state_dict"

        return getattr(self, step_name)

    def __getitem__(self, name: str):
        return self.get(name)

    def __setitem__(self, name: str, value: int):
        name_set = f"n_{name}_steps"
        assert hasattr(self, name_set), f"Key {name_set} missing in state_dict"
        setattr(self, name_set, value)


if __name__ == "__main__":
    counter = StepsCounter(["train", "val"])

    counter.update("train", 1)

    print(counter)
