"""
Multi-focus image fusion dataset

MFF-WHU dataset datapipeline
"""

import os
from contextlib import nullcontext
from copy import deepcopy
from pathlib import Path
from typing import Literal

import kornia.augmentation as K
import numpy as np
import torch as th
import torch.nn.functional as F
from kornia.constants import Resample
from kornia.io import ImageLoadType, load_image
from rich.progress import track
from safetensors.torch import load_file
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms.functional import InterpolationMode, resized_crop

from utils import default, easy_logger, is_main_process

logger = easy_logger(func_name="MFF_WHU")


class MFFWHUDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        transform: list[callable] | None = None,
        transform_ratio: float = 0.0,
        output_size: int = 128,
        get_name: bool = False,
        with_mask: bool = False,
        with_txt: bool = False,
        use_gt: bool = False,
        *,
        mode: Literal[
            "train", "test", "all"
        ] = "test",  # 'all' for all the data to train, like VAE
        device: "th.device | str | None" = None,
        stop_aug_when_n_iters: int = 0,
    ):
        super().__init__()

        self.data_dir = Path(data_dir)
        self.get_name = get_name
        self.with_txt = with_txt
        self.mode = mode
        self.with_mask = with_mask
        self.use_gt = use_gt
        self.stop_aug_when_n_iters = stop_aug_when_n_iters

        _load_type = ImageLoadType.RGB8
        load_keys = {"gt": "full_clear", "far": "source_2", "near": "source_1"}

        if device is not None:
            self.device = th.device(device)
            self.cuda_stream = th.cuda.stream(th.cuda.Stream(device=self.device))
        else:
            self.device = "cpu"
            self.cuda_stream = nullcontext()

        if self.mode in ["train", "test"]:
            file_names = self.data_dir / f"{self.mode}.txt"
            assert file_names.exists(), "file names not found."
            file_names = np.loadtxt(file_names, dtype=str)
        elif self.mode == "all":
            logger.info(f"loading all data in dir {self.data_dir}")
            file_names = os.listdir(os.path.join(self.data_dir, load_keys["far"]))
            # hack the mode to be 'train'
            self.mode = "train"

        # file_names = np.loadtxt(file_names, dtype=str)
        self.far_paths = [self.data_dir / load_keys["far"] / fn for fn in file_names]
        self.near_paths = [self.data_dir / load_keys["near"] / fn for fn in file_names]
        self.gt_paths = [self.data_dir / load_keys["gt"] / fn for fn in file_names]

        # check
        sort_keys = lambda x: x.stem
        self.far_paths.sort(key=sort_keys)
        self.near_paths.sort(key=sort_keys)
        self.gt_paths.sort(key=sort_keys)
        if with_mask:
            self.mask_paths = [
                self.data_dir / "mask" / (fn.split(".")[0] + ".png")
                for fn in file_names
            ]
            self.mask_paths.sort(key=sort_keys)
            assert (
                len(self.far_paths)
                == len(self.near_paths)
                == len(self.gt_paths)
                == len(self.mask_paths)
            ), "number of files are not matched"
            _mask_paths = [mp.stem for mp in self.mask_paths]
            _gt_paths = [gt_p.stem for gt_p in self.gt_paths]
            assert _mask_paths == _gt_paths, "mask and gt are not matched"
            del _mask_paths, _gt_paths
        else:
            assert (
                len(self.far_paths) == len(self.near_paths) == len(self.gt_paths)
            ), "number of files are not matched"

        # load images
        self.far_files, self.near_files, self.gt_files = [], [], []
        for far_p, near_p, gt_p in track(
            zip(self.far_paths, self.near_paths, self.gt_paths),
            description="loading images...",
            total=len(self.far_paths),
            disable=not is_main_process(),
        ):
            self.far_files.append(load_image(far_p, _load_type))
            self.near_files.append(load_image(near_p, _load_type))
            self.gt_files.append(load_image(gt_p, _load_type))

        if with_mask:
            self.mask_files = []
            for mask_p in self.mask_paths:
                self.mask_files.append(load_image(mask_p, ImageLoadType.UNCHANGED)[0])

        logger.info("loaded all images done.")
        logger.info(
            "found {} far/near image pairs for mode {}".format(
                len(self.far_files), mode
            )
        )

        # load txt features
        if self.with_txt:
            txt_file = Path(data_dir) / "t5_feature_MFF-WHU.safetensors"
            assert txt_file.exists(), "txt features not found."
            logger.info("loading txt features from {}".format(txt_file))
            self.txt_features = load_file(txt_file)

        self.output_size = output_size

        # transforms
        self.img_interp_type = Resample.BILINEAR
        self.img_align_corners = (
            True if self.img_interp_type != Resample.NEAREST else None
        )

        if transform is not None or self.mode == "test":
            self.default_transform = None
            self.aug_transform = None
        else:
            data_keys = ["input", "input"]
            if self.use_gt:
                data_keys.append("input")
            if self.with_mask:
                data_keys.append("mask")
            if self.mode == "train":
                if not self.with_txt:
                    _scale = (0.2, 1.0)
                else:
                    _scale = (0.85, 1.0)
                self.K_crop_resize = K.AugmentationSequential(
                    K.RandomResizedCrop(
                        size=(output_size, output_size),
                        ratio=(0.8, 1.4),
                        scale=_scale,
                        keepdim=True,
                        resample=self.img_interp_type,
                        align_corners=self.img_align_corners,
                    ),
                    data_keys=data_keys,
                    same_on_batch=True,
                )
                self.default_transform = K.AugmentationSequential(
                    K.RandomRotation(
                        degrees=(-20, 20),
                        p=transform_ratio,
                        keepdim=True,
                        resample=self.img_interp_type,
                        align_corners=self.img_align_corners,
                    ),
                    K.RandomHorizontalFlip(p=transform_ratio / 2, keepdim=True),
                    K.RandomVerticalFlip(p=transform_ratio / 2, keepdim=True),
                    data_keys=data_keys,
                    same_on_batch=True,
                )
                # self.aug_transform = K.AugmentationSequential(
                #     K.ColorJiggle(0.1, 0.1, 0.1, 0.1, p=transform_ratio / 2, keepdim=True),
                #     K.RandomLinearIllumination(gain=(0.01, 0.1), p=transform_ratio, keepdim=True),
                #     K.RandomMotionBlur(kernel_size=(7, 7), angle=(-30, 30), direction=0.5, p=transform_ratio / 2, keepdim=True),
                #     data_keys=['input', 'input'],  # only apply on under and over images
                #     same_on_batch=True,
                # )
                self.aug_transform = None

        # counter for augmentation
        self.n_iters = 0

    @staticmethod
    def random_crop_resize(
        *imgs,
        output_size,
        interpolation=InterpolationMode.BILINEAR,
        params: tuple | None = None,
    ):
        img_sz = imgs[0].shape[-2:]

        if params is None:
            min_crop_size = output_size

            left = np.random.randint(0, img_sz[1] - min_crop_size + 1)
            top = np.random.randint(0, img_sz[0] - min_crop_size + 1)

            max_width = min(img_sz[1] - left, img_sz[1])
            max_height = min(img_sz[0] - top, img_sz[0])

            width = np.random.randint(min_crop_size, max_width + 1)
            height = np.random.randint(min_crop_size, max_height + 1)
        else:
            left, top, width, height = params

        resized_crop_imgs = []
        for img in imgs:
            _img = resized_crop(
                img,
                top,
                left,
                height,
                width,
                size=(output_size, output_size),
                interpolation=interpolation,
            )
            resized_crop_imgs.append(_img)

        params = (left, top, width, height)

        return resized_crop_imgs, params

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

    def process_imgs(self, *imgs):
        data = {}
        if hasattr(self, "K_crop_resize"):
            imgs = self.K_crop_resize(*imgs)

        if self.mode == "train" and self.n_iters <= self.stop_aug_when_n_iters:
            if self.default_transform is not None:
                imgs = self.default_transform(*imgs)
            if self.aug_transform is not None:
                aug_imgs = self.aug_transform(
                    *imgs[:2]
                )  # first two images are over and under
                imgs[:2] = aug_imgs

        data["far"], data["near"] = imgs[:2]
        _idx = 2
        if self.use_gt:
            data["gt"] = imgs[_idx]
            _idx += 1
        if self.with_mask:
            data["mask"] = imgs[_idx]

        return data

    def __getitem__(self, index):
        with self.cuda_stream:
            far = self.far_files[index].to(self.device).type(th.float32) / 255.0
            near = self.near_files[index].to(self.device).type(th.float32) / 255.0
            if self.use_gt:
                gt = self.gt_files[index].to(self.device).type(th.float32) / 255.0
            if self.with_mask:
                mask = self.mask_files[index].to(self.device).type(th.float32)
            if self.with_txt:
                name = self.far_paths[index].stem
                txt_feature = (
                    self.txt_features[name][0].to(self.device).type(th.float32)
                )

        # augmentations
        aug_imgs = [far, near]
        if self.use_gt:
            aug_imgs.append(gt)
        if self.with_mask:
            aug_imgs.append(mask)
        outp = self.process_imgs(*aug_imgs)

        # organize output
        if not self.use_gt:
            outp["gt"] = th.cat([outp["far"], outp["near"]], dim=0)
        if not self.with_mask:
            outp["mask"] = False
        if self.with_txt:
            outp["txt"] = txt_feature
        if self.get_name:
            file_name = self.far_paths[index].name
            outp["name"] = file_name

        return outp

    def __len__(self):
        return len(self.far_files)


if __name__ == "__main__":
    ds = MFFWHUDataset(
        "/Data3/cao/ZiHanCao/datasets/MFF-WHU/MFI-WHU",
        mode="train",
        transform_ratio=0.3,
        with_mask=True,
        with_txt=True,
        output_size=256,
        get_name=True,
        use_gt=True,
    )
    dl = DataLoader(ds, batch_size=6, shuffle=False, num_workers=0)

    # import accelerate
    # accelerator = accelerate.Accelerator()
    # dl = accelerator.prepare(dl)

    for i, data in enumerate(dl):
        # far, near, mask, gt = data['far'], data['near'], data['mask'], data['gt']
        print(
            data["name"],
            data["far"].shape,
            data["near"].shape,
            data["gt"].shape,
            data["mask"].shape,
        )
