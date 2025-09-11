"""
NVIDIA DALI pipeline for fast loading dataset

Author: Zihan Cao
date: 2024/07/02
"""

import math
import os

## Nvidia DALI pipeline
from pathlib import Path
from typing import Literal, Sequence

import cv2
import numpy as np

# from nvidia.dali.pipeline import Pipeline
# from nvidia.dali.pipeline.experimental import pipeline_def as pipeline_def_experimental
import nvidia.dali.fn as fn
import nvidia.dali.types as types
import torch
from beartype import beartype
from nvidia.dali import pipeline_def
from nvidia.dali.plugin.base_iterator import LastBatchPolicy
from nvidia.dali.plugin.pytorch import DALIGenericIterator
from nvidia.dali.plugin.pytorch.fn import torch_python_function
from safetensors import safe_open
from typeguard import typechecked

rng = np.random.default_rng(seed=2024 + 2)


from utils import easy_logger

logger = easy_logger(func_name="M3FD")


## helper functions


def to_two_tuple(x: "int | float"):
    return (x, x)


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
def string_to_ascii_array(s):
    ascii_values = [ord(char) for char in s]
    return np.array(ascii_values, dtype=np.uint8)


def ascii_tensor_to_string(ascii_tensor):
    ascii_array = ascii_tensor.detach().cpu().numpy()

    string_s = []
    for arr in ascii_array:
        characters = [chr(code) for code in arr]
        string_s.append("".join(characters))

    return string_s


def py_seed():
    # int32_min = np.iinfo(np.int32).min
    # max32_max = np.iinfo(np.int32).max
    return rng.integers(-2147483648, 2147483647)


# external DALI reading pipeline
class M3FDExternalInputCallable:
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
        assert self.base_dir.exists(), f"{self.base_dir} does not exist"

        self.mode = mode
        self.with_mask = with_mask
        self.with_txt_feature = with_txt_feature
        assert mode in ["train", "test"], 'mode must be either "train" or "test"'

        EXTENSIONS = ["jpg", "png", "bmp"]

        self._vi_img_paths = []
        self._ir_img_paths = []
        self._mask_paths = []

        if mode == "train":
            dataset_dir = self.base_dir / "M3FD_Detection"
        else:
            dataset_dir = self.base_dir / "M3FD_Fusion"

        for ext in EXTENSIONS:
            self._vi_img_paths.extend(list((dataset_dir / "vi").glob(f"*.{ext}")))
            self._ir_img_paths.extend(list((dataset_dir / "ir").glob(f"*.{ext}")))
            if with_mask:
                self._mask_paths.extend(list((dataset_dir / "mask").glob(f"*.{ext}")))

        # sort paths
        sorted_key = lambda p: os.path.basename(p).split(".")[0]
        self._vi_img_paths = sorted(self._vi_img_paths, key=sorted_key)
        self._ir_img_paths = sorted(self._ir_img_paths, key=sorted_key)

        # load txt and txt feature
        if with_txt_feature:
            _folder_name = "M3FD_Detection" if mode == "train" else "M3FD_Fusion"
            self.text_feature_vi_f = safe_open(
                self.base_dir / _folder_name / f"t5_feature_M3FD_{mode}_vi.safetensors",
                framework="numpy",
            )
            self.text_feature_ir_f = safe_open(
                self.base_dir / _folder_name / f"t5_feature_M3FD_{mode}_ir.safetensors",
                framework="numpy",
            )
            assert len(self.text_feature_vi_f.keys()) == len(
                self.vi_img_paths
            ), "txt features must have the same size with visible images"
            assert len(self.text_feature_ir_f.keys()) == len(
                self.ir_img_paths
            ), "txt features must have the same size with infrared images"

        vi_names = [p.name.split(".")[0] for p in self._vi_img_paths]
        ir_names = [p.name.split(".")[0] for p in self._ir_img_paths]
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
            self._mask_paths = sorted(self._mask_paths, key=sorted_key)
            mask_names = [p.name.split(".")[0] for p in self._mask_paths]
            assert (
                vi_names == mask_names
            ), "visible, infrared and mask images must have the same name"

        self.total_n_samples = len(self._vi_img_paths)
        assert self.total_n_samples > 0, f"no image found in {dataset_dir}"
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
                0, self.total_n_samples - 1, random_n_samples, dtype=int
            )
            self._vi_img_paths = [self._vi_img_paths[i] for i in _perm]
            self._ir_img_paths = [self._ir_img_paths[i] for i in _perm]
            # names_vi = [p.name.split('.')[0] for p in self._vi_img_paths]
            # names_ir = [p.name.split('.')[0] for p in self._ir_img_paths]
            if with_mask:
                self._mask_paths = [self._mask_paths[i] for i in _perm]
            # if with_txt_feature:
            #     self.text_feature_vi_f = {k: self.text_feature_vi_f.get_tensor(k) for k in names_vi}
            #     self.text_feature_ir_f = {k: self.text_feature_ir_f.get_tensor(k) for k in names_ir}

        self.shard_id = shard_id
        self.num_shards = num_shards
        # If the dataset size is not divisibvle by number of shards, the trailing samples will
        # be omitted.
        self.shard_size = len(self._vi_img_paths) // num_shards
        self.shard_offset = self.shard_size * shard_id
        # If the shard size is not divisible by the batch size, the last incomplete batch
        # will be omitted.
        self.full_iterations = self.shard_size // bs
        self.perm = np.arange(self.total_n_samples)  # permutation of indices
        self.last_seen_epoch = 0
        # (
        #     None  # so that we don't have to recompute the `self.perm` for every sample
        # )

    @property
    def vi_img_paths(self):
        return [str(p) for p in self._vi_img_paths]

    @property
    def ir_img_paths(self):
        return [str(p) for p in self._ir_img_paths]

    @property
    def mask_paths(self):
        return [str(p) for p in self._mask_paths]

    def __call__(self, sample_info):
        sample_idx = sample_info.idx_in_epoch
        # print('sample one from M3FD')
        if sample_info.iteration >= self.full_iterations:
            # print(f'raise: {sample_info.iteration} | {self.full_iterations}')
            raise StopIteration()

        # shuffle
        if self.last_seen_epoch != sample_info.epoch_idx and self.shuffle:
            self.last_seen_epoch = sample_info.epoch_idx
            self.perm = np.random.default_rng(
                seed=42 + sample_info.epoch_idx
            ).permutation(len(self._vi_img_paths))
        sample_idx = self.perm[sample_info.idx_in_epoch]

        vi_img_path, ir_img_path = (
            self._vi_img_paths[sample_idx],
            self._ir_img_paths[sample_idx],
        )
        if self.with_mask:
            mask_path = self._mask_paths[sample_idx]

        # 1. f-read to buffer
        # NOTE: f-read for png images is extremly slow
        with open(vi_img_path, "rb") as f:
            vi_img = np.frombuffer(f.read(), dtype=np.uint8)

        with open(ir_img_path, "rb") as f:
            ir_img = np.frombuffer(f.read(), dtype=np.uint8)

        # 2. opencv to read (speed x2)
        # NOTE: when read jpg file, however opencv is slower than f-read
        # we resort to opencv to read
        # vi_img = cv2.imread(str(vi_img_path), cv2.IMREAD_COLOR)
        # ir_img = cv2.imread(str(ir_img_path), cv2.IMREAD_COLOR)

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
        return len(self._vi_img_paths)


class M3FDDALIPipeLoader:
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

    @typechecked()
    def __init__(
        self,
        base_dir: str,
        mode: Literal["train", "test"],
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

        # if with_mask and mode == 'test':
        #     logger.warning('`with_mask` must be False when `mode` is "test", we set `with_mask` to False by default')
        #     with_mask = False
        self.with_mask = with_mask
        self.get_name = get_name
        self.only_y_component = only_y_component
        if only_y_component:
            logger.warning(
                f"{__class__.__name__}: we only use y component of VIS image"
            )

        self.const = types.Constant(255, dtype=types.UINT8)
        self.crop_strategy = crop_strategy
        self.only_resize = only_resize
        self.with_txt_feature = with_txt_feature
        if crop_strategy == "manual":
            assert isinstance(
                output_size, int
            ), '`output_size` must be an integer when crop strategy is "manual"'

        if only_resize is not None:
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

        # device mapping
        device = str(device)
        if device == "cuda":
            device_id = shard_id
        else:
            device_id = torch.device(device).index
        map_device = "cpu"

        self.external_source = M3FDExternalInputCallable(
            base_dir,
            mode=mode,
            with_mask=with_mask,
            with_txt_feature=with_txt_feature,
            bs=batch_size,
            shard_id=shard_id,
            num_shards=num_shards,
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
        def m3fd_pipeline(
            base_dir: str,
            mode: Literal["train", "test"],
            with_mask: bool,
            with_txt_feature: bool,
            batch_size: int,
            shard_id: int = 0,
            num_shards: int = 1,
            output_size: int = output_size,
            shuffle: bool = False,
            fast_eval_n_samples: int = None,
            const: types.Constant = types.Constant(255, dtype=types.UINT8),
            reduce_label: bool = True,
        ):
            n_output = 2 + sum([with_mask, with_txt_feature, get_name])

            external_source = fn.external_source(
                source=self.external_source,
                num_outputs=n_output,
                batch=False,
            )

            vi, ir = external_source[:2]
            _shape = fn.peek_image_shape(vi)
            vi = fn.decoders.image(vi, device=map_device)
            ir = fn.decoders.image(ir, device=map_device)
            _idx = 2

            # transformations
            is_train = mode == "train"

            # 1. extract y component of vi and ensure ir is gray image
            ir = fn.color_space_conversion(
                ir, image_type=types.RGB, output_type=types.GRAY
            )
            if self.only_y_component:
                vi = fn.color_space_conversion(
                    vi, image_type=types.RGB, output_type=types.GRAY
                )

            if is_train:
                _flip_prob, _hsv_prob = 0.3, 0.2
                if_horizon_flip = fn.random.coin_flip(probability=_flip_prob)
                if_vertical_flip = fn.random.coin_flip(probability=_flip_prob)
                angle = fn.random.uniform(range=(-20, 20))
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

            if not is_train and batch_size != 1:
                # ensure the images in a batch to be the same size
                SIZE = (768, 1024)

                vi = fn.resize(
                    vi,
                    resize_x=SIZE[1],
                    resize_y=SIZE[0],
                    interp_type=self.img_interp_type,
                    antialias=self.img_antialias,
                )
                ir = fn.resize(
                    ir,
                    resize_x=SIZE[1],
                    resize_y=SIZE[0],
                    interp_type=self.img_interp_type,
                    antialias=self.img_antialias,
                )

                # NOTE: the mask image is not loaded in test mode.

            vi = vi.gpu()
            ir = ir.gpu()

            vi = fn.transpose(vi, perm=[2, 0, 1])
            ir = fn.transpose(ir, perm=[2, 0, 1])

            vi = fn.cast(vi, dtype=types.FLOAT) / const
            ir = fn.cast(ir, dtype=types.FLOAT) / const

            ## mask processing
            if with_mask:
                mask = fn.decoders.image(external_source[2], device=map_device)
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
                                random_area=(crop_size[0] / 1024, 1.0),
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
                if not is_train and batch_size != 1:
                    # ensure the images in a batch to be the same size
                    SIZE = (768, 1024)

                    mask = fn.resize(mask, resize_x=SIZE[1], resize_y=SIZE[0])

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
            if with_txt_feature:
                txt_feat = external_source[_idx]
                txt_feat = txt_feat.gpu()
                outp.append(txt_feat)
                _idx += 1
            if get_name:
                outp.append(external_source[_idx])

            return tuple(outp)

        self.pipe = m3fd_pipeline(
            base_dir,
            mode,
            with_mask,
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
        length = len(self.external_source) // (
            self.generic_loader.batch_size * self.num_shards
        )
        if len(self.external_source) % self.generic_loader.batch_size != 0:
            length += 1

        return length


if __name__ == "__main__":
    from tqdm import tqdm

    train_loader = M3FDDALIPipeLoader(
        base_dir="/Data3/cao/ZiHanCao/datasets/VIF-M3FD",
        mode="train",
        with_mask=True,
        with_txt_feature=True,
        batch_size=8,
        shard_id=0,
        num_shards=1,
        output_size=(224, 280),
        n_thread=8,
        fast_eval_n_samples=None,
        crop_strategy="crop_resize",
        get_name=True,
        reduce_label=True,
    )

    for i, data in tqdm(enumerate(train_loader), total=len(train_loader)):
        vi, ir, mask, gt, txt, path = (
            data["vi"],
            data["ir"],
            data["mask"],
            data["gt"],
            data["txt"],
            data["name"],
        )

        # logger.info(data[0].shape)
        pass
    #     # vi, ir, mask, gt, path = data

    #     # logger.debug(ascii_tensor_to_string(path))

    #     # pass

    # print('done')

    # ext_source = M3FDExternalInputCallable(base_dir='/Data3/cao/ZiHanCao/datasets/M3FD',
    #                                         bs=1,
    #                                         mode='train',
    #                                         with_mask=True,
    #                                         get_name=True)

    # from types import SimpleNamespace
    # for i in tqdm(range(3000)):
    #     sample_info = SimpleNamespace(idx_in_epoch=i, iteration=i, epoch_idx=0)
    #     data = ext_source(sample_info)
    #     vi, ir, mask, name = data
    #     # logger.info(vi.shape)
    #     # logger.info(name)

    #     pass
