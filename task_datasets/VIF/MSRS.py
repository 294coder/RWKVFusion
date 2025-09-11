"""
Author: Zihan Cao
Date: 2024-10-09

Copyright (c) 2024 by Zihan Cao, All Rights Reserved.

"""

# ==================== history ======================

# 2024-10-09: add mode=all to load all data from files


# ==================================================

import os
import os.path as osp
import time
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from multiprocessing import Manager, Pool
from pathlib import Path
from typing import Literal, Sequence

import kornia.augmentation as K
import numpy as np
import pandas as pd
import torch
from beartype import beartype
from kornia.constants import Resample
from PIL.Image import open as PIL_open
from rich.progress import (
    BarColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from safetensors import safe_open
from safetensors.torch import load_file
from torch.utils.data import Dataset
from torchvision.transforms import Compose, ToTensor
from torchvision.transforms.functional import to_tensor
from typeguard import typechecked

from utils import easy_logger

logger = easy_logger(func_name="MSRS")


def to_two_tuple(x: "int | float"):
    return (x, x)


def single_process_load_img(
    vi_path: str,
    ir_path: str,
    mask_path: str,
    task_id=None,
    multi_proc_tqdm: bool = False,
    with_mask=False,
    progress: dict = {},
    only_y_component: bool = False,
):
    pil_img_to_tensor = lambda p, mode: to_tensor(PIL_open(p).convert(mode))
    vi_imgs = []
    ir_imgs = []
    mask_imgs = []
    if not with_mask:
        mask_path = ["" for _ in range(len(vi_path))]
    enum_paths = enumerate(zip(vi_path, ir_path, mask_path), 1)
    if not multi_proc_tqdm:
        import tqdm

        tbar = tqdm.tqdm(enum_paths, total=len(vi_path), desc="loading image...")
    else:
        tbar = enum_paths
    for i, (vi, ir, mask) in tbar:
        vi_imgs.append(pil_img_to_tensor(vi, "RGB" if not only_y_component else "L"))
        ir_imgs.append(pil_img_to_tensor(ir, "L"))
        if with_mask:
            mask_imgs.append(pil_img_to_tensor(mask, "L").type(torch.float32))
        progress[task_id] = {"progress": i, "total": len(vi_path)}

    return vi_imgs, ir_imgs, mask_imgs


# NOTE: slow than one process loading, maybe the number of image is too small
def multiprocess_load_img(
    n_proc: int,
    vi_path: str,
    ir_path: str,
    mask_path: str,
    with_mask: bool,
    only_y_component: bool = False,
):
    n_img = len(vi_path)
    logger.info(f"Loading {n_img} images with {n_proc} processes")
    logger.warning(
        "BUGY: multiprocess is much more slower than single process when the number of file is less than 10_000."
    )

    tbar = Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeRemainingColumn(),
        TimeElapsedColumn(),
    )

    n_img_per_proc = n_img // n_proc
    feature = []
    tqdm = True
    with tbar:
        with Manager() as manager:
            _progress = manager.dict()
            with ProcessPoolExecutor(max_workers=n_proc) as executer:
                for i in range(n_proc):
                    start = i * n_img_per_proc
                    end = (i + 1) * n_img_per_proc if i != n_proc - 1 else n_img
                    task_id = tbar.add_task(
                        f"Process {i}: loading images", visible=True
                    )
                    feature.append(
                        executer.submit(
                            single_process_load_img,
                            vi_path[start:end],
                            ir_path[start:end],
                            mask_path[start:end],
                            task_id,
                            tqdm,
                            with_mask,
                            _progress,
                            only_y_component,
                        )
                    )
                # monitor the progress
                while True:
                    all_done = all(f.done() for f in feature)
                    if all_done:
                        logger.info("loading images done")
                        break

                    for task_id, update_data in _progress.items():
                        n_done = update_data["progress"]
                        total = update_data["total"]
                        tbar.update(
                            task_id,
                            total=total,
                            completed=n_done,
                            visible=n_done < total,
                        )
                    time.sleep(0.1)

                # parse results
                vi_imgs = []
                ir_imgs = []
                mask_imgs = []
                for f in feature:
                    vi_img, ir_img, mask_img = f.result()
                    vi_imgs.extend(vi_img)
                    ir_imgs.extend(ir_img)
                    mask_imgs.extend(mask_img)

    return vi_imgs, ir_imgs, mask_imgs


class MSRSDatasets(Dataset):
    def __init__(
        self,
        dir_path: str,
        mode: Literal["train", "test", "detection", "all"] = "train",
        output_size: int = 72,
        transform_ratio: bool = 0.0,
        transforms: "list[torch.nn.Module] | None" = None,
        load_to_ram: bool = True,
        n_proc_load: int = 1,
        get_name: bool = False,
        reduce_label: bool = True,
        with_mask: bool = True,
        only_y_component: bool = False,
        with_txt_feature: bool = False,
        only_resize: "Sequence[int, int] | None" = None,
        fast_eval_n_samples: "int | None" = None,  # used in `mode=test`
    ):
        super().__init__()
        self.dir_path = dir_path
        self.mode = mode
        self.load_to_ram = load_to_ram
        self.n_proc_load = n_proc_load
        self.get_name = get_name
        self.logger = logger
        self.with_mask = with_mask
        self.reduce_label = reduce_label
        self.only_y_component = only_y_component
        self.with_txt_feature = with_txt_feature
        self.fast_eval_n_samples = fast_eval_n_samples
        assert (
            fast_eval_n_samples is None or mode == "test"
        ), '`fast_eval_n_samples` can not be set when `mode` is not "test"'
        if fast_eval_n_samples is not None:
            logger.info("fast eval n samples: ", fast_eval_n_samples)
        # assert not (mode == 'all' and with_txt_feature), '`with_txt_feature` must be False when `mode` is "all"'

        if with_txt_feature and only_resize is None:
            logger.info(
                "we only resize images to size (280, 224) for when using txt feature"
            )
            self.only_resize = (224, 280)
            assert (
                n_proc_load == 1
            ), "when using txt feature, we only support single process loading"
        else:
            self.only_resize = only_resize
            logger.info(f"only resize images to {self.only_resize}")

        if only_y_component:
            logger.warning(
                f"{__class__.__name__}: we only use y component of VIS image"
            )
        if reduce_label:
            logger.info(
                f"{__class__.__name__}: note that we reduce label larger than 1 to be 1"
            )
        _default_crop_size = to_two_tuple(output_size)
        assert n_proc_load >= 1 and isinstance(
            n_proc_load, int
        ), "n_proc_load must be int number and be greater than or equal to 1"
        assert self.mode in [
            "train",
            "test",
            "detection",
            "all",
        ], 'mode must be either "train", "test", "detection" or "all"'

        # get image paths
        self.collect_img_paths()
        if self.with_mask:
            assert len(self.vi_paths) == len(self.ir_paths) == len(self.mask_paths)
        else:
            assert len(self.vi_paths) == len(self.ir_paths)

        # load txt features
        if self.with_txt_feature:
            self.load_txt_feature()

        # fast test n samples
        if fast_eval_n_samples is not None:
            _perm = np.linspace(
                0, len(self.vi_paths) - 1, fast_eval_n_samples, dtype=np.int32
            )
            self.vi_paths = [self.vi_paths[i] for i in _perm]
            self.ir_paths = [self.ir_paths[i] for i in _perm]
            if self.with_mask:
                self.mask_paths = [self.mask_paths[i] for i in _perm]
            if self.with_txt_feature:
                _txt_feature_vi = {}
                _txt_feature_ir = {}
                for p in self.vi_paths:
                    stem = p.stem
                    # append in
                    _txt_feature_vi[stem] = self.txt_feature_vi.pop(stem)
                    _txt_feature_ir[stem] = self.txt_feature_ir.pop(stem)
                self.txt_feature_vi = _txt_feature_vi
                self.txt_feature_ir = _txt_feature_ir

        # load images
        self.load_imgs()

        # get image/mask augmentations
        resample_type = Resample.BILINEAR
        if transforms is None and mode == "train":
            if not self.only_resize:
                self.transforms = [
                    K.RandomResizedCrop(
                        _default_crop_size, scale=(0.8, 1), resample=resample_type
                    ),
                    K.RandomHorizontalFlip(p=transform_ratio),
                    K.RandomVerticalFlip(p=transform_ratio),
                    K.ColorJiggle(0.1, 0.1, 0.1, 0.1, p=transform_ratio)
                    if only_y_component
                    else None,
                    K.RandomRotation(20, p=transform_ratio, resample=resample_type),
                ]
            else:
                self.transforms = [K.Resize(self.only_resize, antialias=False)]
        elif transforms is not None and mode == "train":
            self.transforms = transforms
        else:
            self.transforms = []

        _transform_flags = [t.flags if t is not None else None for t in self.transforms]
        self.mask_trans_flags = self.prepare_mask_transforme_flags(_transform_flags)

    def prepare_mask_transforme_flags(self, transform_flags: list[dict]):
        mask_flags = deepcopy(transform_flags)
        for idx, f in enumerate(transform_flags):
            if f is None:
                continue
            if "align_corners" in f:
                align_coners = f
                mask_flags[idx]["align_corners"] = None
            if "resample_method" in f:
                resample_method = f["resample_method"]
                mask_flags[idx]["resample_method"] = Resample.get("nearest")
            if "resample" in f:
                resample = f["resample"]
                mask_flags[idx]["resample"] = Resample.get("nearest")

        return mask_flags

    def collect_img_paths(self):
        scan_dir = Path(self.dir_path)
        self.vi_paths, self.ir_paths, self.mask_paths = [], [], []

        if self.mode == "train":
            scan_dir_path = scan_dir / "train" / "vi"
            self.vi_paths = list(scan_dir_path.glob("*.jpg"))

            scan_dir_path = scan_dir / "train" / "ir"
            self.ir_paths = list(scan_dir_path.glob("*.jpg"))

        elif self.mode == "detection":
            scan_dir_path = scan_dir / "detection" / "vi"
            self.vi_paths = list(scan_dir_path.glob("*.png"))

            scan_dir_path = scan_dir / "detection" / "ir"
            self.ir_paths = list(scan_dir_path.glob("*.png"))

        elif self.mode == "all":
            for mode in ["train", "test", "detection"]:
                _scan_dir_path = scan_dir / mode / "vi"
                self.vi_paths.extend(list(_scan_dir_path.glob("*")))

                _scan_dir_path = scan_dir / mode / "ir"
                self.ir_paths.extend(list(_scan_dir_path.glob("*")))

        elif self.mode == "test":
            scan_dir_path = scan_dir / "test" / "vi"
            self.vi_paths = list(scan_dir_path.glob("*"))

            scan_dir_path = scan_dir / "test" / "ir"
            self.ir_paths = list(scan_dir_path.glob("*"))

        sorted_key = lambda p: p.stem
        self.ir_paths = sorted(self.ir_paths, key=sorted_key)
        self.vi_paths = sorted(self.vi_paths, key=sorted_key)

        ir_names = [p.stem for p in self.ir_paths]
        vi_names = [p.stem for p in self.vi_paths]

        assert (
            vi_names == ir_names
        ), "visible and infrared images must have the same name"

        # collect mask paths
        modes = [self.mode] if self.mode != "all" else ["train", "test", "detection"]
        for mode in modes:
            scan_dir_path = scan_dir / str(mode) / "Segmentation_labels"
            if not Path(scan_dir_path).exists():
                logger.warning(
                    f"mask is not exists of mode {self.mode}, set `with_mask=False`"
                )
                self.with_mask = False
            else:
                self.mask_paths.extend(list(scan_dir_path.glob("*.png")))
        self.mask_paths = sorted(self.mask_paths, key=sorted_key)

        # check names
        assert (
            len(self.vi_paths) == len(self.ir_paths) == len(self.mask_paths)
        ), "visible, infrared and mask images must have the same length"
        mask_names = [p.stem for p in self.mask_paths]
        assert (
            ir_names == vi_names == mask_names
        ), "visible, infrared and mask images must have the same names"

    def load_imgs(self):
        if self.load_to_ram:
            logger.info(f"Loading {len(self.vi_paths)} image pairs to RAM ...")
            if self.n_proc_load > 1:
                self.vi_imgs, self.ir_imgs, self.mask_imgs = multiprocess_load_img(
                    self.n_proc_load,
                    self.vi_paths,
                    self.ir_paths,
                    self.mask_paths,
                    self.with_mask,
                    self.only_y_component,
                )
            else:
                self.vi_imgs, self.ir_imgs, self.mask_imgs = single_process_load_img(
                    self.vi_paths,
                    self.ir_paths,
                    self.mask_paths,
                    with_mask=self.with_mask,
                    only_y_component=self.only_y_component,
                )
            if self.with_mask:
                assert len(self.vi_imgs) == len(self.ir_imgs) == len(self.mask_imgs)
            else:
                assert len(self.vi_imgs) == len(self.ir_imgs)
        else:
            logger.info("Loading images on-the-fly")

    def load_txt_feature(self):
        modes = [self.mode] if self.mode != "all" else ["train", "test", "detection"]
        vi_txt_features = []
        ir_txt_features = []
        for mode in modes:
            logger.info(f"load t5 encoded txt features of {mode} ...")
            vi_txt_features.append(
                load_file(
                    Path(self.dir_path)
                    / mode
                    / f"t5_feature_MSRS_{mode}_vi.safetensors",
                    device="cpu",
                )
            )
            ir_txt_features.append(
                load_file(
                    Path(self.dir_path)
                    / mode
                    / f"t5_feature_MSRS_{mode}_ir.safetensors",
                    device="cpu",
                )
            )
        logger.info("loading t5 encoded txt features done")

        # merge txt features
        self.txt_feature_vi = {}
        self.txt_feature_ir = {}
        for vi_txt_features, ir_txt_features in zip(vi_txt_features, ir_txt_features):
            for vi_key, vi_tf in vi_txt_features.items():
                self.txt_feature_vi[vi_key] = vi_tf
            for ir_key, ir_tf in ir_txt_features.items():
                self.txt_feature_ir[ir_key] = ir_tf

    def __len__(self):
        return len(self.vi_paths)

    def apply_transforms(self, vi_img, ir_img, mask_img):
        # vi_img, ir_img = map(self.to_tensor, (vi_img, ir_img))
        if self.with_mask:
            mask_img = torch.from_numpy(np.array(mask_img, dtype=np.float32))

        # cast People class to 1 not 2
        # because in MSRS, there is not label (but original MFNet has).
        if self.reduce_label and self.with_mask:
            mask_img[mask_img > 1.0] = 1.0

        if self.mode == "train":
            for t, m_f in zip(self.transforms, self.mask_trans_flags):
                if t is not None:
                    vi_img = t(vi_img)
                    params = t._params
                    if not isinstance(t, K.ColorJiggle):
                        ir_img = t(ir_img, params=params)
                    if self.with_mask and not isinstance(t, K.ColorJiggle):
                        mask_img = t(mask_img[None], params=params, **m_f)
                        mask_img = mask_img.squeeze()

            return vi_img[0], ir_img[0], mask_img[None] if self.with_mask else mask_img
        else:
            return vi_img, ir_img, mask_img

    def __getitem__(self, index):
        with_mask = self.with_mask

        if self.load_to_ram:
            vi_img = self.vi_imgs[index]
            ir_img = self.ir_imgs[index]
            if with_mask:
                mask_img = self.mask_imgs[index]
            else:
                mask_img = False
        else:
            vi_img = to_tensor(PIL_open(self.vi_paths[index]))
            ir_img = to_tensor(PIL_open(self.ir_paths[index]))
            if with_mask:
                mask_img = to_tensor(PIL_open(self.mask_paths[index]))
            else:
                mask_img = False

        vi_img, ir_img, mask_img = self.apply_transforms(vi_img, ir_img, mask_img)

        outp = {"vi": vi_img, "ir": ir_img, "gt": torch.cat([vi_img, ir_img], dim=0)}
        if self.with_mask:
            outp["mask"] = mask_img

        if self.with_txt_feature:
            name = Path(self.vi_paths[index]).stem
            vi_txt_feat = self.txt_feature_vi[name][0].to(torch.float32)
            ir_txt_feat = self.txt_feature_ir[name][0].to(torch.float32)
            outp["txt"] = torch.cat([vi_txt_feat, ir_txt_feat], dim=-1)
        if self.get_name:
            outp["name"] = self.vi_paths[index].name

        return outp


## DALI Pipeline
from pathlib import Path
from typing import Literal

import numpy as np
import nvidia.dali.fn as fn
import nvidia.dali.types as types
from nvidia.dali import pipeline_def
from nvidia.dali.plugin.base_iterator import LastBatchPolicy
from nvidia.dali.plugin.pytorch import DALIGenericIterator
from nvidia.dali.plugin.pytorch.fn import torch_python_function

rng = np.random.default_rng(seed=2024 + 1)


## helper functions


def _print_shape(x):
    print(x.shape)
    return x


def _mask_to_01(cls, value):
    def _inner(mask):
        mask[mask == cls] = value

        return mask

    return _inner


def _mask_gr_0to1(mask):
    mask[mask > 0] = 1

    return mask


## handle the string of file name


# to ascii code
def string_to_ascii_array(s, max_len=None):
    ascii_values = [ord(char) for char in s]
    if max_len is None:
        return np.array(ascii_values, dtype=np.uint8)
    else:
        arr = np.zeros(max_len, dtype=np.uint8)
        if len(ascii_values) > max_len:
            ascii_values = ascii_values[:max_len]
        arr[: len(ascii_values)] = ascii_values
        return arr


def ascii_tensor_to_string(ascii_tensor):
    # batched tensor of ascii code

    if ascii_tensor.ndim == 1:
        ascii_tensor = ascii_tensor[None]
    if torch.is_tensor(ascii_tensor):
        ascii_array = ascii_tensor.detach().cpu().numpy()

    string = []
    for arr in ascii_array:
        characters = [chr(code) for code in arr]
        string.append("".join(characters))

    return string if len(string) > 1 else string[0]


def py_seed():
    # int32_min = np.iinfo(np.int32).min
    # max32_max = np.iinfo(np.int32).max
    return rng.integers(-2147483648, 2147483647)


# external DALI reading pipeline
class MSRSExternalInputCallable:
    # @typechecked()
    @beartype
    def __init__(
        self,
        base_dir: str,
        mode: Literal["train", "test", "all"],
        with_mask: bool,
        with_txt: bool,
        with_txt_feature: bool,
        bs: int,
        shard_id: int = 0,
        num_shards: int = 1,
        shuffle: bool = True,
        random_n_samples: int | None = None,
        get_name: bool = False,
        load_txt_feat_in_ram: bool = False,
    ):
        self.bs = bs
        self.shuffle = shuffle
        self.shard_id = shard_id
        self.get_name = get_name
        self.load_txt_feat_in_ram = load_txt_feat_in_ram

        # handle the input data
        self.base_dir = Path(base_dir)
        self.mode = mode
        self.with_mask = with_mask
        self.with_txt = with_txt
        self.with_txt_feature = with_txt_feature
        assert mode in [
            "train",
            "test",
            "all",
        ], 'mode must be either "train" or "test" or "all"'
        if mode == "all":
            load_modes = ["train", "test"]
            assert (
                not with_txt_feature
            ), '`with_txt_feature` must be False when `mode` is "all"'
        else:
            load_modes = [mode]

        EXTENSIONS = ["jpg", "png", "bmp"]

        self.vi_img_paths = []
        self.ir_img_paths = []
        self.mask_paths = []

        for mode in load_modes:
            for ext in EXTENSIONS:
                self.vi_img_paths.extend(
                    list((self.base_dir / mode / "vi").glob(f"*.{ext}"))
                )
                self.ir_img_paths.extend(
                    list((self.base_dir / mode / "ir").glob(f"*.{ext}"))
                )
                if with_mask:
                    self.mask_paths.extend(
                        list(
                            (self.base_dir / mode / "Segmentation_labels").glob(
                                f"*.{ext}"
                            )
                        )
                    )

        # sort paths and lengths, names assertions
        sorted_key = lambda p: p.name.split(".")[0]
        self.vi_img_paths = sorted(self.vi_img_paths, key=sorted_key)
        self.ir_img_paths = sorted(self.ir_img_paths, key=sorted_key)

        # load txt and txt feature
        if with_txt:
            self.txt_vi = pd.read_csv(
                self.base_dir / mode / f"caption_MSRS_{mode}_vi.csv"
            )
            self.txt_ir = pd.read_csv(
                self.base_dir / mode / f"caption_MSRS_{mode}_ir.csv"
            )
            self.txt_vi = self.txt_vi.set_index(self.txt_vi.columns[0])
            self.txt_ir = self.txt_ir.set_index(self.txt_ir.columns[0])
            assert len(self.txt_vi) == len(
                self.txt_ir
            ), "txt files must have the same size"
            assert len(self.txt_vi) == len(
                self.vi_img_paths
            ), "txt files must have the same size"
            assert len(self.txt_ir) == len(
                self.ir_img_paths
            ), "txt files must have the same size"
        if with_txt_feature:
            # load feature in memory
            if load_txt_feat_in_ram:
                logger.info(f"loading t5 encoded txt features ...")
                self.txt_feature_vi = load_file(
                    self.base_dir / mode / f"t5_feature_MSRS_{mode}_vi.safetensors",
                    device=shard_id,
                )
                self.txt_feature_ir = load_file(
                    self.base_dir / mode / f"t5_feature_MSRS_{mode}_ir.safetensors",
                    device=shard_id,
                )
                assert len(self.txt_feature_vi) == len(
                    self.vi_img_paths
                ), "txt features must have the same size with visible images"
                assert len(self.txt_feature_ir) == len(
                    self.ir_img_paths
                ), "txt features must have the same size with infrared images"
                logger.info(f"loading t5 encoded txt features done")
            # load it to GPU on-the-fly
            else:
                self.txt_feature_vi_f = safe_open(
                    self.base_dir / mode / f"t5_feature_MSRS_{mode}_vi.safetensors",
                    framework="numpy",
                )
                self.txt_feature_ir_f = safe_open(
                    self.base_dir / mode / f"t5_feature_MSRS_{mode}_ir.safetensors",
                    framework="numpy",
                )
                assert len(self.txt_feature_vi_f.keys()) == len(
                    self.vi_img_paths
                ), "txt features must have the same size with images"
                assert len(self.txt_feature_ir_f.keys()) == len(
                    self.ir_img_paths
                ), "txt features must have the same size with images"

        vi_names = [p.name.split(".")[0] for p in self.vi_img_paths]
        ir_names = [p.name.split(".")[0] for p in self.ir_img_paths]
        assert (
            vi_names == ir_names
        ), "visible and infrared images must have the same name"
        assert (
            len(self.vi_img_paths) == len(self.ir_img_paths)
        ), f"visible and infrared images must have the same size, but found vi: {len(self.vi_img_paths)} and ir: {len(self.ir_img_paths)}"

        if with_mask:
            self.mask_paths = sorted(self.mask_paths, key=sorted_key)
            assert (
                len(self.vi_img_paths) == len(self.mask_paths)
            ), f"visible and mask images must have the same size, but found vi: {len(self.vi_img_paths)} and mask: {len(self.mask_paths)}"
            mask_names = [p.name.split(".")[0] for p in self.mask_paths]
            assert (
                vi_names == mask_names
            ), "visible, infrared and mask images must have the same name"

        self.total_n_samples = len(self.vi_img_paths)
        logger.info(
            f'{mode=}, found {self.total_n_samples} vi/ir/{"mask" if with_mask else ""} pairs'
        )

        # randomly select samples
        if random_n_samples is not None:
            assert (
                bs <= random_n_samples
            ), "`bs` must be less than or equal to `random_n_samples`"
            logger.warning(
                f"select {random_n_samples} samples randomly. only used for a fast evaluation"
            )

            # FIXME: random select it in different processes, may cause different samples in each process
            # _perm = np.random.permutation(self.total_n_samples)
            _perm = np.linspace(
                0, self.total_n_samples - 1, random_n_samples, dtype=np.int32
            )
            self.vi_img_paths = [self.vi_img_paths[i] for i in _perm[:random_n_samples]]
            self.ir_img_paths = [self.ir_img_paths[i] for i in _perm[:random_n_samples]]
            names_vi = [osp.basename(p).split(".")[0] for p in self.vi_img_paths]
            names_ir = [osp.basename(p).split(".")[0] for p in self.ir_img_paths]
            if with_mask:
                self.mask_paths = [self.mask_paths[i] for i in _perm[:random_n_samples]]
            if with_txt:
                # sort by stem
                self.txt_vi = self.txt_vi.loc[names_vi]
                self.txt_ir = self.txt_ir.loc[names_ir]
            # if with_txt_feature:
            #     if load_txt_feat_in_ram:
            #         self.txt_feature_vi = {k: self.txt_feature_vi[k] for k in names_vi}
            #         self.txt_feature_ir = {k: self.txt_feature_ir[k] for k in names_ir}

        self.shard_id = shard_id
        self.num_shards = num_shards
        # If the dataset size is not divisibvle by number of shards, the trailing samples will
        # be omitted.
        self.shard_size = len(self.vi_img_paths) // num_shards
        self.shard_offset = self.shard_size * shard_id
        # If the shard size is not divisible by the batch size, the last incomplete batch
        # will be omitted.
        self.full_iterations = self.shard_size // bs
        self.perm = np.arange(self.total_n_samples)  # permutation of indices
        self.last_seen_epoch = 0
        # (
        #     None  # so that we don't have to recompute the `self.perm` for every sample
        # )

    def __call__(self, sample_info):
        sample_idx = sample_info.idx_in_epoch
        # print('sample one from MSRS')
        if sample_info.iteration >= self.full_iterations:
            # print(f'raise: {sample_info.iteration} | {self.full_iterations}')
            raise StopIteration()

        # shuffle
        if self.last_seen_epoch != sample_info.epoch_idx and self.shuffle:
            self.last_seen_epoch = sample_info.epoch_idx
            self.perm = np.random.default_rng(
                seed=42 + sample_info.epoch_idx
            ).permutation(len(self.vi_img_paths))
        sample_idx = self.perm[sample_info.idx_in_epoch]

        vi_img_path, ir_img_path = (
            self.vi_img_paths[sample_idx],
            self.ir_img_paths[sample_idx],
        )
        if self.with_mask:
            mask_path = self.mask_paths[sample_idx]

        with open(vi_img_path, "rb") as f:
            vi_img = np.frombuffer(f.read(), dtype=np.uint8)

        with open(ir_img_path, "rb") as f:
            ir_img = np.frombuffer(f.read(), dtype=np.uint8)

        outp = [vi_img, ir_img]
        if self.with_mask:
            with open(mask_path, "rb") as f:
                mask = np.frombuffer(f.read(), dtype=np.uint8)
            outp.append(mask)

        vi_file_name = Path(vi_img_path).stem
        ir_file_name = Path(ir_img_path).stem
        if self.with_txt:
            vi_txt = self.txt_vi.loc[vi_file_name].caption
            ir_txt = self.txt_ir.loc[ir_file_name].caption
            vi_ascii = string_to_ascii_array(vi_txt, max_len=512)
            ir_ascii = string_to_ascii_array(ir_txt, max_len=512)
            outp.append(vi_ascii)
            outp.append(ir_ascii)

        if self.with_txt_feature:
            if self.load_txt_feat_in_ram:
                vi_txt_feat = self.txt_feature_vi[vi_file_name][0].type(torch.float32)
                ir_txt_feat = self.txt_feature_ir[ir_file_name][0].type(torch.float32)
                txt_feat = torch.cat([vi_txt_feat, ir_txt_feat], dim=-1)
            else:
                vi_txt_feat = self.txt_feature_vi_f.get_tensor(vi_file_name)[0].astype(
                    np.float32
                )
                ir_txt_feat = self.txt_feature_ir_f.get_tensor(ir_file_name)[0].astype(
                    np.float32
                )
                txt_feat = np.concatenate([vi_txt_feat, ir_txt_feat], axis=-1)
            outp.append(txt_feat)

        if self.get_name:
            outp.append(string_to_ascii_array(vi_img_path.name))

        # vi, ir, mask, vi_txt, ir_txt, txt_feat, name
        return tuple(outp)

    def __len__(self):
        return len(self.vi_img_paths)


class MSRSDALIPipeLoader:
    """
    LLVIP dataset using DALI pipeline

    speed test:
        CPU: Intel(R) Core(TM) i9-10900X CPU @ 3.70GHz
        GPU: NVIDIA GeForce RTX 3090

        Config:
            `batch_size`: 32
            `num_thread`: 8
            `n_shards`: 1
            `output_size`: 256

        Speed (when stable): ~7it/s v.s. ~2it/s (using `torch.utils.data.DataLoader` with multiprocess).
    """

    # @typechecked
    @beartype
    def __init__(
        self,
        base_dir: str,
        mode: Literal["train", "test", "all"],
        output_size: int | Sequence[int] | None = None,
        with_mask: bool = True,
        with_txt: bool = False,
        with_txt_feature: bool = False,
        batch_size: int = 32,
        shard_id: int = 0,
        num_shards: int = 1,
        n_thread: int = 8,
        device: "str | torch.device" = "cuda",
        shuffle: bool = False,
        fast_eval_n_samples: int | None = None,
        get_name: bool = False,
        reduce_label: bool = True,
        crop_strategy: Literal["manual", "crop", "crop_resize"] = "crop",
        only_y_component: bool = False,  # only use y component of vi image
        only_resize: Sequence[int]
        | None = None,  # recommand to be (384, 288) with ratio 1.3333, thus the heigh and width are divisible by 16
        load_txt_feat_in_ram: bool = True,
    ):
        self.shard_id = shard_id
        self.num_shards = num_shards
        self.batch_size = batch_size

        if reduce_label:
            logger.info(f"{__class__}: note that we reduce label larger than 1 to be 1")

        self.with_mask = with_mask
        self.with_txt = with_txt
        self.with_txt_feat = with_txt_feature
        assert (
            output_size is not None or mode == "test"
        ), 'output_size must be specified when mode is "train"'
        if crop_strategy == "manual":
            assert isinstance(
                output_size, int
            ), '`output_size` must be an integer when crop strategy is "manual"'

        self.get_name = get_name
        self.only_y_component = only_y_component
        if only_y_component:
            logger.warning(
                f"{__class__.__name__}: we only use y component of VIS image"
            )

        self.const = types.Constant(255, dtype=types.UINT8)
        self.crop_strategy = crop_strategy
        self.only_resize = only_resize

        if only_resize is None:
            logger.info(
                f"`only_resize_size` is specified, we will resize the images to the specified size and ignore the `crop_strategy`"
            )
        else:
            logger.info(f"crop strategy: {self.crop_strategy}")

        device = str(device)
        if device == "cuda":
            device_id = 0
        else:
            device_id = torch.device(device).index

        map_device = "cpu"

        self.img_interp_type = types.DALIInterpType.INTERP_LINEAR
        self.img_antialias = False

        @pipeline_def(
            batch_size=batch_size,
            num_threads=n_thread,
            device_id=device_id,
            enable_conditionals=True,
        )
        def msrs_pipeline(
            base_dir: str,
            mode: Literal["train", "test", "all"],
            with_mask: bool,
            with_txt: bool,
            with_txt_feat: bool,
            batch_size: int,
            shard_id: int = 0,
            num_shards: int = 1,
            output_size: int | Sequence[int] = output_size,
            shuffle: bool = False,
            fast_eval_n_samples: int = None,
            const: types.Constant = types.Constant(255, dtype=types.UINT8),
            reduce_label: bool = True,
        ):
            self.external_source = MSRSExternalInputCallable(
                base_dir,
                mode,
                with_mask,
                with_txt,
                with_txt_feat,
                batch_size,
                shard_id,
                num_shards,
                shuffle=shuffle,
                random_n_samples=fast_eval_n_samples,
                get_name=get_name,
                load_txt_feat_in_ram=load_txt_feat_in_ram,
            )
            self.n_samples = len(self.external_source)
            n_output = 2 + sum([with_mask, with_txt, with_txt, with_txt_feat, get_name])

            external_source = fn.external_source(
                source=self.external_source,
                num_outputs=n_output,
                batch=False,
            )

            vi, ir = external_source[0], external_source[1]
            _idx = 2
            _shape = fn.peek_image_shape(vi)
            vi = fn.decoders.image(vi, device=map_device)
            ir = fn.decoders.image(ir, device=map_device)

            # transformations
            is_train = mode == "train" or mode == "all"

            # 1. extract y component of vi and ensure ir is gray image
            ir = fn.color_space_conversion(
                ir, image_type=types.RGB, output_type=types.GRAY
            )
            if self.only_y_component:
                vi = fn.color_space_conversion(
                    vi, image_type=types.RGB, output_type=types.GRAY
                )

            if is_train:
                _flip_prob, _hsv_prob = 0.3, 0.3
                if_horizon_flip = fn.random.coin_flip(probability=_flip_prob)
                if_vertical_flip = fn.random.coin_flip(probability=_flip_prob)
                angle = fn.random.uniform(range=(-30, 30))
                brightness1 = fn.random.uniform(range=(0.4, 1.6))
                brightness2 = fn.random.uniform(range=(0.4, 1.6))
                # if_hsv = fn.random.coin_flip(probability=_hsv_prob)
                # hue, saturation = fn.random.uniform(range=(0, 360)), fn.random.uniform(range=(0, 4))

                # only resize
                if self.only_resize is not None:
                    vi = fn.resize(
                        vi,
                        resize_x=self.only_resize[1],
                        resize_y=self.only_resize[0],
                        interp_type=self.img_interp_type,
                        antialias=self.img_antialias,
                    )
                    ir = fn.resize(
                        ir,
                        resize_x=self.only_resize[1],
                        resize_y=self.only_resize[0],
                        interp_type=self.img_interp_type,
                        antialias=self.img_antialias,
                    )
                else:
                    if self.crop_strategy == "manual":
                        # NOTE: `manual` crop strategy can just crop rectangle region

                        # compute the min side of the image
                        h = fn.slice(_shape, 0, 1, axes=(0,), device="cpu")
                        w = fn.slice(_shape, 1, 1, axes=(0,), device="cpu")
                        _max_size = fn.cast(
                            fn.reductions.min(fn.stack(h, w), device="cpu"),
                            dtype=types.INT32,
                        )
                        _min_size = types.Constant(output_size, dtype=types.INT32)
                        _crop_range = fn.cast(
                            fn.stack(_min_size, _max_size, device="cpu"),
                            dtype=types.FLOAT,
                        )
                        crop_size = fn.random.uniform(
                            range=_crop_range, dtype=types.FLOAT
                        )
                        crop_size = fn.cast(
                            fn.stack(crop_size, crop_size), dtype=types.FLOAT
                        )
                        crop_pos_x = fn.random.uniform(range=(0.0, 1.0))
                        crop_pos_y = fn.random.uniform(range=(0.0, 1.0))
                    elif self.crop_strategy == "crop":
                        crop_size = (
                            (output_size, output_size)
                            if isinstance(output_size, int)
                            else output_size
                        )
                        crop_pos_x = fn.random.uniform(range=(0.0, 1.0))
                        crop_pos_y = fn.random.uniform(range=(0.0, 1.0))
                    elif self.crop_strategy == "crop_resize":
                        crop_size = (
                            (output_size, output_size)
                            if isinstance(output_size, int)
                            else output_size
                        )
                        random_cr_seed = py_seed()

                    ## apply transformations to vi and ir same

                    # 2. crop via different strategies
                    if self.crop_strategy == "crop_resize":
                        vi = fn.random_resized_crop(
                            vi,
                            size=crop_size,
                            random_area=(0.8, 1.0),
                            random_aspect_ratio=(0.8, 1.2),
                            interp_type=self.img_interp_type,
                            seed=random_cr_seed,
                        )
                        ir = fn.random_resized_crop(
                            ir,
                            size=crop_size,
                            random_area=(0.8, 1.0),
                            random_aspect_ratio=(0.8, 1.2),
                            interp_type=self.img_interp_type,
                            seed=random_cr_seed,
                        )
                    elif self.crop_strategy == "manual":
                        vi = fn.crop(
                            vi,
                            crop=crop_size,
                            crop_pos_x=crop_pos_x,
                            crop_pos_y=crop_pos_y,
                        )
                        ir = fn.crop(
                            ir,
                            crop=crop_size,
                            crop_pos_x=crop_pos_x,
                            crop_pos_y=crop_pos_y,
                        )
                        vi = fn.resize(
                            vi,
                            resize_x=output_size,
                            resize_y=output_size,
                            interp_type=self.img_interp_type,
                            antialias=self.img_antialias,
                        )
                        ir = fn.resize(
                            ir,
                            resize_x=output_size,
                            resize_y=output_size,
                            interp_type=self.img_interp_type,
                            antialias=self.img_antialias,
                        )
                    elif self.crop_strategy == "crop":
                        vi = fn.crop(
                            vi,
                            crop=crop_size,
                            crop_pos_x=crop_pos_x,
                            crop_pos_y=crop_pos_y,
                        )
                        ir = fn.crop(
                            ir,
                            crop=crop_size,
                            crop_pos_x=crop_pos_x,
                            crop_pos_y=crop_pos_y,
                        )

                # # 3. other augmentations
                # vi = fn.brightness(vi, brightness=brightness1)
                # ir = fn.brightness(ir, brightness=brightness2)

                # # 4. rotation
                # vi = fn.rotate(vi, angle=angle, keep_size=True)
                # ir = fn.rotate(ir, angle=angle, keep_size=True)

                # # 5. flip
                # vi = fn.flip(vi, horizontal=if_horizon_flip, vertical=if_vertical_flip)
                # ir = fn.flip(ir, horizontal=if_horizon_flip, vertical=if_vertical_flip)

                # 6. hsv
                # if if_hsv:
                #     vi = fn.hsv(vi, hue=hue, saturation=saturation, value=1)

            vi = vi.gpu()
            ir = ir.gpu()

            vi = fn.transpose(vi, perm=[2, 0, 1])
            ir = fn.transpose(ir, perm=[2, 0, 1])

            vi = fn.cast(vi, dtype=types.FLOAT) / const
            ir = fn.cast(ir, dtype=types.FLOAT) / const

            if with_mask:
                mask = fn.decoders.image(external_source[_idx], device=map_device)
                _idx += 1
                mask = mask.gpu()
                mask = fn.color_space_conversion(
                    mask, image_type=types.RGB, output_type=types.GRAY
                )

                if is_train:
                    # 1. only_resize or cropping
                    if self.only_resize is not None:
                        mask = fn.resize(
                            mask,
                            resize_x=self.only_resize[1],
                            resize_y=self.only_resize[0],
                            interp_type=types.DALIInterpType.INTERP_NN,
                            antialias=False,
                        )
                    else:
                        if self.crop_strategy == "manual":
                            mask = fn.crop(
                                mask,
                                crop=crop_size,
                                crop_pos_x=crop_pos_x,
                                crop_pos_y=crop_pos_y,
                            )
                            mask = fn.resize(
                                mask,
                                resize_x=output_size,
                                resize_y=output_size,
                                interp_type=types.DALIInterpType.INTERP_NN,
                                antialias=False,
                            )
                        elif self.crop_strategy == "crop_resize":
                            mask = fn.random_resized_crop(
                                mask,
                                size=crop_size,
                                random_area=(0.8, 1.0),
                                random_aspect_ratio=(0.8, 1.2),
                                seed=random_cr_seed,
                                interp_type=types.DALIInterpType.INTERP_NN,
                                antialias=False,
                            )
                        elif self.crop_strategy == "crop":
                            mask = fn.crop(
                                mask,
                                crop=crop_size,
                                crop_pos_x=crop_pos_x,
                                crop_pos_y=crop_pos_y,
                            )

                    # # 2. rotate
                    # mask = fn.rotate(mask, angle=angle, keep_size=True, interp_type=types.DALIInterpType.INTERP_NN)

                    # # 3. flip
                    # mask = fn.flip(mask, horizontal=if_horizon_flip, vertical=if_vertical_flip)

                mask = fn.cast(mask, dtype=types.FLOAT)

                # set element > 0 (is an object) to be 1
                if reduce_label:
                    mask = torch_python_function(mask, function=_mask_gr_0to1)

                mask = fn.transpose(mask, perm=[2, 0, 1])
                mask = fn.cast_like(mask, vi)

            # organize the output
            outp = [vi, ir]
            if with_mask:
                outp.append(mask)
            outp.append(fn.cat(vi, ir, axis=0))  # gt
            if with_txt:
                vi_txt = external_source[_idx]
                ir_txt = external_source[_idx + 1]
                _idx += 2
                outp.append(vi_txt)
                outp.append(ir_txt)
            if with_txt_feat:
                txt_feat = external_source[_idx]
                txt_feat = txt_feat.gpu()
                outp.append(txt_feat)
                _idx += 1
            if get_name:
                outp.append(external_source[_idx])

            return tuple(outp)

        self.pipe = msrs_pipeline(
            base_dir,
            mode,
            with_mask,
            with_txt,
            with_txt_feature,
            batch_size,
            shard_id,
            num_shards,
            output_size=output_size,
            const=self.const,
            shuffle=shuffle,
            fast_eval_n_samples=fast_eval_n_samples,
            reduce_label=reduce_label,
        )
        self.pipe.build()

        loader_output_map = ["vi", "ir", "gt"]
        if with_mask:
            loader_output_map = ["vi", "ir", "mask", "gt"]
        if with_txt:
            loader_output_map.append("vi_txt")
            loader_output_map.append("ir_txt")
        if with_txt_feature:
            loader_output_map.append("txt")
        if get_name:
            loader_output_map.append("name")

        self.generic_loader = DALIGenericIterator(
            self.pipe,
            output_map=loader_output_map,
            auto_reset=True,
            last_batch_padded=True,
            last_batch_policy=LastBatchPolicy.FILL,
        )

    def __iter__(self):
        return self

    def __next__(self) -> dict[str, torch.Tensor]:
        data = next(self.generic_loader)[0]

        # outputs include: vi, ir, mask, gt, vi_txt, ir_txt, txt_feat, name
        return data

    def __len__(self):
        length = len(self.external_source.vi_img_paths) // (
            self.generic_loader.batch_size * self.num_shards
        )
        if len(self.external_source.vi_img_paths) % self.generic_loader.batch_size != 0:
            length += 1

        return length


if __name__ == "__main__":
    # import time
    from torch.utils.data import DataLoader

    # # t1 = time.time()

    # def collate_fn(batch):
    #     vi_batch = []
    #     ir_batch = []
    #     for b in batch:
    #         vi_batch.append(b['vi'])
    #         ir_batch.append(b['ir'])
    #     return torch.stack(vi_batch), torch.stack(ir_batch)

    ds = MSRSDatasets(
        dir_path="/Data3/cao/ZiHanCao/datasets/VIF-MSRS",
        mode="all",
        load_to_ram=True,
        n_proc_load=1,
        output_size=256,
        transform_ratio=1.0,
        get_name=False,
        reduce_label=True,
        only_y_component=False,
        with_mask=True,
        with_txt_feature=True,
        fast_eval_n_samples=None,
    )
    loader = DataLoader(ds, batch_size=3, shuffle=True, num_workers=2)

    from tqdm import tqdm

    for data in tqdm(loader, total=len(loader)):
        # print(data.keys())  # data['vi']
        pass

    # def get_train_input(train_dataloader):
    #     def _iter_two_inputs(batch):
    #         yield from batch

    #     _loader = iter(train_dataloader)
    #     while True:
    #         try:
    #             batch = next(_loader)
    #             yield from _iter_two_inputs(batch)
    #         except StopIteration:
    #             print('reloading the dataloader')
    #             import time
    #             time.sleep(2)
    #             _loader = iter(train_dataloader)
    #             batch = next(_loader)
    #             yield from _iter_two_inputs(batch)

    # for data in get_train_input(loader):
    #     print(data.shape)

    ################################# check pipeline ################################
    # from tqdm import tqdm
    # torch.cuda.set_device(1)
    # # import matplotlib.pyplot as plt

    # train_loader = MSRSDALIPipeLoader(base_dir='/Data3/cao/ZiHanCao/datasets/VIF-MSRS', mode='train', with_mask=True,
    #                                    batch_size=32, shard_id=0, num_shards=1, output_size=(224, 280), n_thread=8, with_txt=False,
    #                                    with_txt_feature=True, shuffle=True, fast_eval_n_samples=None, crop_strategy='crop_resize',
    #                                    get_name=True, only_y_component=False)

    # for i, data in tqdm(enumerate(train_loader), total=len(train_loader)):
    #     vi, ir, mask, gt, txt = data['vi'], data['ir'], data['mask'], data['gt'], data['txt']
    # print(vi.shape, ir.shape)
    # assert data is not None
    # print(data['vi'].device, data['ir'].device, data['mask'].device, data['gt'].device, data['txt'].device)
