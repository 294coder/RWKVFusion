"""
LLVIP datasets
Dataset paper: https://arxiv.org/pdf/2108.10831

Code by: Zihan Cao
Date: 2024/06/17
"""

# note: load all to RAM consume lots of memory, maybe load on-the-fly is better

import math
import os
import os.path as osp
import queue
import time
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Manager, Pool, Process, current_process
from pathlib import Path
from typing import Literal, Sequence

import cv2
import kornia
import numpy as np

# from nvidia.dali.pipeline.experimental import pipeline_def
import nvidia.dali.fn as fn
import nvidia.dali.types as types
import torch
from beartype import beartype
from kornia.augmentation import (
    RandomHorizontalFlip,
    RandomResizedCrop,
    RandomRotation,
    RandomVerticalFlip,
)
from nvidia.dali import pipeline_def
from nvidia.dali.plugin.base_iterator import LastBatchPolicy
from nvidia.dali.plugin.pytorch import DALIGenericIterator
from nvidia.dali.plugin.pytorch.fn import torch_python_function
from PIL.Image import open as PIL_open
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from safetensors import safe_open
from torch.utils.data import Dataset, Sampler
from torchvision.transforms import Compose
from torchvision.transforms.functional import to_tensor
from typeguard import typechecked

from utils import EasyProgress, easy_logger

logger = easy_logger(func_name="LLVIP")


def to_two_tuple(x: "int | float"):
    return (x, x)


def single_process_load_img(
    vi_path: str,
    ir_path: str,
    task_id=None,
    multi_proc_tqdm: bool = False,
    progress: dict = {},
):
    # img reading function
    cv2_img_to_tensor_rgb = lambda p: to_tensor(
        cv2.cvtColor(cv2.imread(p, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    )
    cv2_img_to_tensor_gray = lambda p: to_tensor(cv2.imread(p, cv2.IMREAD_GRAYSCALE))
    pil_img_to_tensor = lambda p: to_tensor(PIL_open(p))

    vi_imgs = []
    ir_imgs = []
    # mask_imgs = []
    enum_paths = enumerate(zip(vi_path, ir_path), 1)
    if not multi_proc_tqdm:
        # import tqdm
        # tbar = tqdm.tqdm(enum_paths, total=len(vi_path), desc='loading image...')
        tbar, task_id = EasyProgress.easy_progress(
            ["loading image..."], [len(vi_path)], is_main_process=True
        )
        iter_list = tbar.track(enum_paths, task_id=task_id)
        tbar.start()
    else:
        iter_list = enum_paths

    for i, (vi, ir) in iter_list:
        vi_imgs.append(cv2_img_to_tensor_rgb(vi))
        ir_imgs.append(cv2_img_to_tensor_gray(ir))
        # mask_imgs.append(torch.tensor(np.array(PIL_open(mask)), dtype=torch.float32))
        progress[task_id] = {"progress": i, "total": len(vi_path)}

    return vi_imgs, ir_imgs  # , mask_imgs


# NOTE: slow than one process loading, maybe the number of image is too small
# TODO: launching lots of processes is much slower, try to use multiprocessing.ThreadPool and its imap fucntion to async the loading procedure
# from multiprocessing.pool import ThreadPool


def multiprocess_load_img(n_proc: int, vi_path: str, ir_path: str):
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
        console=logger._console,
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
                            task_id,
                            tqdm,
                            _progress,
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
                for f in feature:
                    vi_img, ir_img = f.result()
                    vi_imgs.extend(vi_img)
                    ir_imgs.extend(ir_img)

    return vi_imgs, ir_imgs


class LLVIPDatasets(Dataset):
    def __init__(
        self,
        dir_path: str,  # dir path for LLVIP dataset
        mode: Literal["train", "test"] = "train",  # only set to 'train' or 'test'
        crop_size: int = 72,  # we need to crop the image to avoid any OOM problem in the model
        transform_ratio: bool = 0.0,  # transform (augmentation) ratio
        transforms: list[
            torch.nn.Module, callable
        ] = None,  # any callable function to transform the vi and ir images
        load_to_ram: bool = True,  # `True` for loading to RAM and `False` for on-the-fly loading
        device: "str | torch.device" = None,
        ram_n_proc_load: int = 1,
        ### NOT RECOMMANDED !!
        # just for fun, do not set them for your good mind
        on_the_fly_n_proc: int = None,
        load_by_queue: bool = False,
        # hybrid loading
        loading_raito: float = 0.0,
    ):
        super().__init__()
        self.dir_path = dir_path
        self.mode = mode
        self.crop_size = crop_size
        self.load_to_ram = load_to_ram
        self.n_proc_load = ram_n_proc_load
        self.device = device  # only use when on-the-fly loading

        ##
        self.on_the_fly_n_proc = on_the_fly_n_proc  # only use when on-the-fly loading
        self.load_by_queue = load_by_queue  # only use when on-the-fly loading
        ##

        # hybrid loading, `loading_ratio` first on RAM and the rest on-the-fly
        self.hybrid_loading = loading_raito > 0
        if self.hybrid_loading:
            assert (
                (not load_by_queue) and (on_the_fly_n_proc is None)
            ), "you must NOT set `load_by_queue` or `on_the_fly_n_proc` when using hybrid loading"
            assert False, "not implemented yet"

        self.logger = logger
        _default_crop_size = to_two_tuple(crop_size)
        assert ram_n_proc_load >= 1 and isinstance(
            ram_n_proc_load, int
        ), "n_proc_load must be int number and be greater than or equal to 1"
        assert self.mode in ["train", "test"], 'mode must be either "train" or "test"'

        ## NOT RECOMMANDED !!
        if load_by_queue:  # this is a toy, you can try by yourself, anyway, it's not recommended and do not waste your time
            logger.error(
                "BUGGY: `load_by_queue` is slow than directly loading, you can try to test the speed, here raise an error"
            )
            # raise RuntimeError

        self.collect_img_paths()
        assert len(self.vi_paths) == len(self.ir_paths)
        self.load_imgs()

        if transforms is None and mode == "train":
            self.transforms = [
                RandomResizedCrop(_default_crop_size),
                RandomHorizontalFlip(p=transform_ratio),
                RandomVerticalFlip(p=transform_ratio),
                RandomRotation(45, p=transform_ratio),
            ]
        elif transforms is not None and mode == "train":
            self.transforms = Compose(transforms)
        else:
            self.transforms = []

        # cuda stream when not load_to_ram
        if not load_to_ram:
            assert device is not None, "device must be specified"
            if load_by_queue:
                assert (
                    on_the_fly_n_proc is not None
                ), "on_the_fly_proc must be specified"
                self.queue = Manager().Queue(
                    100
                )  # the Manager.Queue is shared, and I guess maybe this affect the speed
                self.prefetch_proc = None
                self.cuda_stream = torch.cuda.Stream(device=device)
            else:
                self.cuda_stream = torch.cuda.Stream(device=device)

    def collect_img_paths(self):
        scan_dir = Path(self.dir_path)

        scan_dir_path = scan_dir / "visible" / self.mode
        self.vi_paths = list(scan_dir_path.glob("*.jpg"))[:400]

        scan_dir_path = scan_dir / "infrared" / self.mode
        self.ir_paths = list(scan_dir_path.glob("*.jpg"))[:400]

    def load_imgs(self):
        if self.load_to_ram:
            self.logger.warning("log all image to RAM. Be careful of memory usage")
            if self.n_proc_load > 1:
                self.vi_imgs, self.ir_imgs = multiprocess_load_img(
                    self.n_proc_load, self.vi_paths, self.ir_paths
                )
            else:
                self.vi_imgs, self.ir_imgs = single_process_load_img(
                    self.vi_paths, self.ir_paths
                )
            assert len(self.vi_imgs) == len(self.ir_imgs)  # == len(self.mask_imgs)
        else:
            self.logger.info("loading images on-the-fly")

        self.logger.info(f"found {len(self.vi_paths)} images")

    def __len__(self):
        return len(self.vi_paths)

    @staticmethod
    def apply_transforms(vi_img, ir_img, mode, transforms):
        if mode == "train":
            for t in transforms:
                vi_img = t(vi_img)
                params = t._params
                ir_img = t(ir_img, params=params)
                # mask_img = t(mask_img, params=params)

            return vi_img[0], ir_img[0]  # , mask_img[0]
        else:
            return vi_img, ir_img  # , mask_img

    @staticmethod
    def _on_the_fly_loading(
        index,
        vi_paths,
        ir_paths,
        queue,
        check_tensor,
        apply_transforms,
        mode,
        transforms,
        device,
    ):
        try:
            vi_img = PIL_open(vi_paths[index])
            ir_img = PIL_open(ir_paths[index])
            vi_img = check_tensor(
                vi_img
            )  # .to(device, non_blocking=True)  # buggy: `to` device show use spawn method and a different process to the same device cause an error
            ir_img = check_tensor(ir_img)  # .to(device, non_blocking=True)
            vi_img, ir_img = apply_transforms(vi_img, ir_img, mode, transforms)

            queue.put((vi_img, ir_img))
        except Exception as e:
            logger.error(
                f"Error when loading image {index} in {current_process().name}, {e}"
            )

    @staticmethod
    def _queue_loading(
        indices: list[int],
        on_the_fly_n_proc,
        on_the_fly_loading,
        vi_paths,
        ir_paths,
        queue,
        check_tensor,
        apply_transforms,
        mode,
        transforms,
        device,
    ):
        with Pool(on_the_fly_n_proc) as pool:
            pool.starmap_async(
                on_the_fly_loading,
                [
                    (
                        index,
                        vi_paths,
                        ir_paths,
                        queue,
                        check_tensor,
                        apply_transforms,
                        mode,
                        transforms,
                        device,
                    )
                    for index in indices
                ],
            ).wait()

            # just for debugging

            # for index in indices:
            #     pool.apply(on_the_fly_loading, args=(index, vi_paths, ir_paths, queue, check_tensor,
            #                                                apply_transforms, mode, transforms, device))

    def start_queue(self, indices: list[int]):
        if not self.load_by_queue:
            return

        if self.prefetch_proc is not None:
            self.prefetch_proc.join()
        self.prefetch_proc = Process(
            target=self._queue_loading,
            args=(
                indices,
                self.on_the_fly_n_proc,
                self._on_the_fly_loading,
                self.vi_paths,
                self.ir_paths,
                self.queue,
                self.check_tensor,
                self.apply_transforms,
                self.mode,
                self.transforms,
                self.device,
            ),
        )
        self.prefetch_proc.start()

    @staticmethod
    def check_tensor(img):
        if not isinstance(img, torch.Tensor):
            img = to_tensor(img)

        return img

    def __getitem__(self, index):
        if self.load_to_ram:
            vi_img = self.vi_imgs[index]
            ir_img = self.ir_imgs[index]
            vi_img, ir_img = self.apply_transforms(
                vi_img, ir_img, self.mode, self.transforms
            )
        else:
            # v1
            if self.load_by_queue:  # may meet any speed problem
                try:
                    vi_img, ir_img = self.queue.get(timeout=10)
                    with torch.cuda.stream(self.cuda_stream):
                        vi_img = vi_img.to(self.device, non_blocking=True)
                        ir_img = ir_img.to(self.device, non_blocking=True)

                    self.cuda_stream.synchronize()

                except queue.Empty:
                    raise StopIteration("Data loading process has finished.")
            # v2# we often choose this
            else:
                with torch.cuda.stream(self.cuda_stream):
                    # vi_img = PIL_open(self.vi_paths[index])
                    # ir_img = PIL_open(self.ir_paths[index])
                    vi_img = cv2.cvtColor(
                        cv2.imread(self.vi_paths[index], cv2.IMREAD_COLOR),
                        cv2.COLOR_BGR2RGB,
                    )
                    ir_img = cv2.imread(self.ir_paths[index], cv2.IMREAD_GRAYSCALE)
                    vi_img, ir_img = map(self.check_tensor, [vi_img, ir_img])
                    vi_img, ir_img = self.apply_transforms(
                        vi_img, ir_img, self.mode, self.transforms
                    )
                    vi_img = vi_img.to(self.device, non_blocking=True)
                    ir_img = ir_img.to(self.device, non_blocking=True)

                self.cuda_stream.synchronize()

            # v3
            # TODO: may load a part of the dataset to RAM and the rest are choosed to using on-the-fly loading

        return vi_img, ir_img, torch.cat([vi_img, ir_img], dim=0)

    def _close(self):
        assert not self.load_to_ram, "only on-the-fly loading need to close"
        if self.prefetch_proc is not None:
            self.prefetch_proc.join()
        self.pool.close()
        self.pool.join()


class ShuffleSampler(Sampler):
    def __init__(self, data_source):
        self.data_source = data_source
        self.indices = list(range(len(data_source)))

    def __iter__(self):
        np.random.shuffle(self.indices)
        return iter(self.indices)

    def __len__(self):
        return len(self.data_source)


### DALI dataloader


rng = np.random.default_rng(seed=2024)

## handle the string of file name


# to ascii code
def string_to_ascii_array(s):
    ascii_values = [ord(char) for char in s]
    return np.array(ascii_values, dtype=np.uint8)


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
class LLVIPExternalInputCallable:
    # @typechecked()
    @beartype
    def __init__(
        self,
        base_dir: str,
        mode: Literal["train", "test", "all"],
        with_mask: bool,
        with_txt_feature: bool,
        bs: int,
        shard_id: int = 0,
        num_shards: int = 1,
        shuffle: bool = True,
        random_n_samples: int | None = None,
        get_name: bool = False,
    ):
        self.bs = bs
        self.shuffle = shuffle
        self.shard_id = shard_id
        self.num_shards = num_shards
        self.get_name = get_name

        # handle the input data
        self.base_dir = Path(base_dir)
        self.mode = mode
        self.with_mask = with_mask
        self.with_txt_feature = with_txt_feature
        assert mode in [
            "train",
            "test",
            "all",
        ], 'mode must be either "train" or "test" or "all"'
        if mode == "all":
            assert (
                not with_txt_feature
            ), '`with_txt_feature` must be False when `mode` is "all"'
        load_modes = [mode] if mode != "all" else ["train", "test"]

        EXTENSIONS = ["jpg", "png", "bmp"]

        self.vi_img_paths = []
        self.ir_img_paths = []
        self.mask_paths = []

        for mode in load_modes:
            for ext in EXTENSIONS:
                self.vi_img_paths.extend(
                    list((self.base_dir / "visible" / mode).glob(f"*.{ext}"))
                )
                self.ir_img_paths.extend(
                    list((self.base_dir / "infrared" / mode).glob(f"*.{ext}"))
                )
                if with_mask:
                    self.mask_paths.extend(
                        list((self.base_dir / "mask" / mode).glob(f"*.{ext}"))
                    )

        # sort paths
        sorted_key = lambda p: os.path.basename(p).split(".")[0]
        self.vi_img_paths = sorted(self.vi_img_paths, key=sorted_key)
        self.ir_img_paths = sorted(self.ir_img_paths, key=sorted_key)

        # load txt and txt feature
        if with_txt_feature:
            self.text_feature_vi_f = safe_open(
                self.base_dir / f"t5_feature_LLVIP_{mode}_vi.safetensors",
                framework="numpy",
            )
            self.text_feature_ir_f = safe_open(
                self.base_dir / f"t5_feature_LLVIP_{mode}_ir.safetensors",
                framework="numpy",
            )
            assert len(self.text_feature_vi_f.keys()) == len(
                self.vi_img_paths
            ), "txt features must have the same size with visible images"
            assert len(self.text_feature_ir_f.keys()) == len(
                self.ir_img_paths
            ), "txt features must have the same size with infrared images"

        vi_names = [p.name.split(".")[0] for p in self.vi_img_paths]
        ir_names = [p.name.split(".")[0] for p in self.ir_img_paths]
        assert (
            vi_names == ir_names
        ), "visible and infrared images must have the same name"

        assert (
            len(self.vi_img_paths) == len(self.ir_img_paths)
        ), f"visible and infrared images must have the same size, but found vi: {len(self.vi_img_paths)} and ir: {len(self.ir_img_paths)}"
        if with_mask:
            assert (
                len(self.vi_img_paths) == len(self.mask_paths)
            ), f"visible and mask images must have the same size, but found vi: {len(self.vi_img_paths)} and mask: {len(self.mask_paths)}"
            self.mask_paths = sorted(self.mask_paths, key=sorted_key)
            mask_names = [p.name.split(".")[0] for p in self.mask_paths]
            assert (
                vi_names == mask_names
            ), "visible, infrared and mask images must have the same name"

        self.total_n_samples = len(self.vi_img_paths)
        logger.info(
            f'{mode=}, found {self.total_n_samples} vi/ir{"/mask" if with_mask else ""} pairs'
        )

        if random_n_samples is not None:
            assert (
                bs <= random_n_samples
            ), "`bs` must be less than or equal to `random_n_samples`"
            logger.warning(
                f"select {random_n_samples} samples randomly. only used for a fast evaluation"
            )
            _perm = np.linspace(
                0, self.total_n_samples - 1, random_n_samples, dtype=np.int32
            )
            self.vi_img_paths = [self.vi_img_paths[i] for i in _perm[:random_n_samples]]
            self.ir_img_paths = [self.ir_img_paths[i] for i in _perm[:random_n_samples]]
            # names_vi = [osp.basename(p).split('.')[0] for p in self.vi_img_paths]
            # names_ir = [osp.basename(p).split('.')[0] for p in self.ir_img_paths]
            if with_mask:
                self.mask_paths = [self.mask_paths[i] for i in _perm[:random_n_samples]]
            # if with_txt_feature:
            #     self.text_feature_vi_f = {k: self.text_feature_vi_f.get_slice(k) for k in names_vi}
            #     self.text_feature_ir_f = {k: self.text_feature_ir_f.get_slice(k) for k in names_ir}

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
        # print('sample one from LLVIP')
        if sample_info.iteration >= self.full_iterations:
            # print(f'raise: {sample_info.iteration} | {self.full_iterations}')
            raise StopIteration()

        # shuffle
        if self.last_seen_epoch != sample_info.epoch_idx and self.shuffle:
            self.last_seen_epoch = sample_info.epoch_idx
            # print(sample_idx.epoch_idx)
            self.perm = np.random.default_rng(
                seed=42 + sample_info.epoch_idx
            ).permutation(len(self.vi_img_paths))
        sample_idx = self.perm[sample_idx]
        # print(sample_info.idx_in_epoch, sample_info.epoch_idx)

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

        if self.with_txt_feature:
            vi_txt_feat = self.text_feature_vi_f.get_tensor(Path(vi_img_path).stem)[
                0
            ].astype(np.float32)
            ir_txt_feat = self.text_feature_ir_f.get_tensor(Path(ir_img_path).stem)[
                0
            ].astype(np.float32)
            txt_feat = np.concatenate([vi_txt_feat, ir_txt_feat], axis=-1)
            outp.append(txt_feat)

        if self.get_name:
            outp.append(string_to_ascii_array(vi_img_path.name))

        return tuple(outp)

    def __len__(self):
        return len(self.vi_img_paths)


def _print_shape(x):
    print(x.shape)
    return x


def _print_value(x):
    print(x)
    return x


def _mask_to_01(cls, value):
    def _inner(mask):
        mask[mask == cls] = value

        return mask

    return _inner


def _mask_gr_0to1(mask):
    mask[mask > 0] = 1

    return mask


class LLVIPDALIPipeLoader:
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

    @beartype
    def __init__(
        self,
        base_dir: str,
        mode: Literal["train", "test", "all"],
        output_size: int | Sequence[int] | None = None,
        with_mask: bool = True,
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
        only_resize: "Sequence | None" = None,  # recommand to be (384, 288) with ratio 1.3333, thus the heigh and width are divisible by 16
    ):
        self.num_shards = num_shards

        if reduce_label:
            logger.info(f"{__class__}: note that we reduce label larger than 1 to be 1")

        self.with_mask = with_mask
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

        assert (
            output_size is not None or mode == "test"
        ), 'output_size must be specified when mode is "train"'
        if mode == "test" and batch_size != 1:
            logger.warning(
                "should be aware of the not same image size in a same batch, "
                "we choose to resize the images with small size to be the biggest size in the batch"
            )

        device = str(device)
        if device == "cuda":
            device_id = shard_id
        else:
            device_id = torch.device(device).index

        map_device = "mixed"

        self.external_source = LLVIPExternalInputCallable(
            base_dir,
            mode,
            with_mask,
            with_txt_feature,
            batch_size,
            shard_id,
            num_shards,
            shuffle=shuffle,
            random_n_samples=fast_eval_n_samples,
            get_name=get_name,
        )
        self.n_samples = len(self.external_source)
        self.img_interp_type = types.DALIInterpType.INTERP_LINEAR
        self.img_antialias = False

        @pipeline_def(
            batch_size=batch_size,
            num_threads=n_thread,
            device_id=device_id,
            enable_conditionals=True,
        )
        def llvip_pipeline(
            mode: Literal["train", "test"],
            with_mask: bool,
            bs: int,
            output_size: int | Sequence[int] = output_size,
            const: types.Constant = types.Constant(255, dtype=types.UINT8),
            reduce_mask: bool = True,
        ):
            n_output = 2 + sum([with_mask, with_txt_feature, get_name])
            external_source = fn.external_source(
                source=self.external_source,
                num_outputs=n_output,
                batch=False,
            )

            vi = fn.decoders.image(external_source[0], device=map_device)
            ir = fn.decoders.image(external_source[1], device=map_device)
            _idx = 2

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
                ## prepare configs for transformations
                _flip_prob, _hsv_prob = 0.3, 0.3
                if_horizon_flip = fn.random.coin_flip(probability=_flip_prob)
                if_vertical_flip = fn.random.coin_flip(probability=_flip_prob)
                angle = fn.random.uniform(range=(-30, 30))
                brightness1 = fn.random.uniform(range=(0.4, 1.6))
                brightness2 = fn.random.uniform(range=(0.4, 1.6))
                # if_hsv = fn.random.coin_flip(probability=_hsv_prob)
                # hue, saturation = fn.random.uniform(range=(0, 360)), fn.random.uniform(range=(0, 4))

                ## apply transformations to vi and ir same

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
                # cropping strategy
                else:
                    if self.crop_strategy == "manual":
                        _max_size = 1024.0
                        _min_size = output_size
                        _crop_size = fn.random.uniform(
                            range=(_min_size, _max_size), dtype=types.FLOAT
                        )
                        crop_size = fn.stack(_crop_size, _crop_size)
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
                        random_cr_seed = (
                            py_seed()
                        )  # fn.random.uniform(range=(0, 1e5), dtype=types.INT32)

                    # 3. crop via different strategies
                    if self.crop_strategy == "crop_resize":
                        vi = fn.random_resized_crop(
                            vi,
                            size=crop_size,
                            random_area=(0.8, 1.0),
                            random_aspect_ratio=(0.8, 1.2),
                            interp_type=self.img_interp_type,
                            antialias=self.img_antialias,
                            seed=random_cr_seed,
                        )
                        ir = fn.random_resized_crop(
                            ir,
                            size=crop_size,
                            random_area=(0.8, 1.0),
                            random_aspect_ratio=(0.8, 1.2),
                            interp_type=self.img_interp_type,
                            antialias=self.img_antialias,
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

                # # 4. other augmentations
                # vi = fn.brightness(vi, brightness=brightness1)
                # ir = fn.brightness(ir, brightness=brightness2)

                # # 5. rotation
                # vi = fn.rotate(vi, angle=angle, keep_size=True)
                # ir = fn.rotate(ir, angle=angle, keep_size=True)

                # # 6. flip
                # vi = fn.flip(vi, horizontal=if_horizon_flip, vertical=if_vertical_flip)
                # ir = fn.flip(ir, horizontal=if_horizon_flip, vertical=if_vertical_flip)

                # 7. hsv
                # if if_hsv:
                # vi = fn.hsv(vi, hue=hue, saturation=saturation, value=1)

            if not is_train and bs != 1:
                # ensure the images in a batch to be the same size
                SIZE = (1024, 1280)

                vi = fn.resize(vi, resize_x=SIZE[1], resize_y=SIZE[0])
                ir = fn.resize(ir, resize_x=SIZE[1], resize_y=SIZE[0])

            vi = vi.gpu()
            ir = ir.gpu()

            vi = fn.transpose(vi, perm=[2, 0, 1])
            ir = fn.transpose(ir, perm=[2, 0, 1])

            vi = fn.cast(vi, dtype=types.FLOAT) / const
            ir = fn.cast(ir, dtype=types.FLOAT) / const

            if with_mask:
                mask = fn.decoders.image(external_source[_idx], device=map_device)
                _idx += 1
                mask = fn.color_space_conversion(
                    mask, image_type=types.RGB, output_type=types.GRAY
                )
                mask = mask.gpu()

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

                # NOTE: the current mask data is only count by numbers of objects, not represents any cls infomation

                # set element > 0 (is an object) to be 1
                if reduce_mask:
                    mask = torch_python_function(mask, function=_mask_gr_0to1)

                mask = fn.transpose(mask, perm=[2, 0, 1])
                mask = fn.cast_like(mask, vi)

            # organize the output
            outp = [vi, ir]
            if with_mask:
                outp.append(mask)
            outp.append(fn.cat(vi, ir, axis=0))  # gt
            if with_txt_feature:
                txt_feat = external_source[_idx]
                txt_feat = txt_feat.gpu()
                outp.append(txt_feat)
                _idx += 1
            if get_name:
                outp.append(external_source[_idx])

            return tuple(outp)

        self.pipe = llvip_pipeline(
            mode=mode,
            with_mask=with_mask,
            bs=batch_size,
            output_size=output_size,
            const=self.const,
            reduce_mask=reduce_label,
        )
        self.pipe.build()

        loader_output_map = ["vi", "ir", "gt"]
        if with_mask:
            loader_output_map = ["vi", "ir", "mask", "gt"]
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

    def __next__(self):
        data = next(self.generic_loader)[0]

        return data

    def __len__(self):
        length = len(self.external_source.vi_img_paths) // (
            self.generic_loader.batch_size * self.num_shards
        )
        if len(self.external_source.vi_img_paths) % self.generic_loader.batch_size != 0:
            length += 1

        return length


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    from tqdm import tqdm

    # t1 = time.time()
    ds = LLVIPDatasets(
        dir_path="/Data3/cao/ZiHanCao/datasets/VIF-LLVIP/data",
        mode="train",
        load_to_ram=False,
        crop_size=256,
        ram_n_proc_load=1,
        device="cuda:0",
        load_by_queue=False,
        on_the_fly_n_proc=1,
    )
    # t2 = time.time()
    # print(f'use time {t2-t1}s')

    dl = DataLoader(ds, batch_size=32, shuffle=False, num_workers=4)
    # dl.start_prefetch = ds.start_queue
    # sampler = ShuffleSampler(ds)

    for vi, ir, gt in tqdm(dl):
        # print(vi.shape)
        pass

    # for ep in range(2):
    #     indices = list(sampler)
    #     dl.start_prefetch(indices)  # call when setting `load_by_queue`

    #     for vi, ir, gt in tqdm(dl):
    #         print(vi.shape)

    ## test pipeline
    # from tqdm import tqdm

    # train_loader = LLVIPDALIPipeLoader(base_dir='/Data3/cao/ZiHanCao/datasets/VIF-LLVIP/data', mode='train', with_mask=True, with_txt_feature=False,
    #                                    batch_size=32, shard_id=1, num_shards=1, output_size=(288, 384), n_thread=8,
    #                                    fast_eval_n_samples=None, get_name=True, reduce_label=False, crop_strategy='crop_resize')

    # for i, data in tqdm(enumerate(train_loader), total=len(train_loader)):
    #     # vi, ir, mask, gt, txt, path = data['vi'], data['ir'], data['mask'], data['gt'], data['txt'], data['name']
    #     vi, ir, gt = data['vi'], data['ir'], data['gt']

    # print('done')

    ## test safetensor
    # d = safe_open('/Data3/cao/ZiHanCao/datasets/LLVIP/data/t5_feature_LLVIP_train_ir.safetensors', framework='numpy')
    # d2 = safe_open('/Data3/cao/ZiHanCao/datasets/LLVIP/data/t5_feature_LLVIP_train_vi.safetensors', framework='numpy')

    # for k in tqdm(d.keys(), desc='loading t5 features', total=len(list(d.keys()))):
    #     tensor = d.get_tensor(k)
    #     tensor2 = d2.get_tensor(k)
