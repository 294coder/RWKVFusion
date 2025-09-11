"""
Multi-exposure image fusion dataset

MEFB dataset datapipeline
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

logger = easy_logger(func_name="MEFB")


class MEFBDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        transform: "list[callable] | None" = None,
        transform_ratio: float = 0.0,
        output_size: int = 128,
        only_y: bool = False,
        with_mask: bool = False,
        with_txt: bool = False,
        get_name: bool = False,
        *,
        mode: Literal[
            "train", "test", "all"
        ] = "train",  # 'all' for all the data to train, like VAE
        device: "th.device | str | None" = None,
        stop_aug_when_n_iters: int = -1,
    ):
        super().__init__()

        self.data_dir = Path(data_dir)
        self.only_y = only_y
        self.get_name = get_name
        self.mode = mode
        self.with_txt = with_txt
        self.with_mask = with_mask
        self.stop_aug_when_n_iters = stop_aug_when_n_iters
        # the aspect ratio of this dataset varies,
        # if with txt feature input into model, make output size larger than 256
        if self.with_txt:
            logger.info(
                "[green underline]with txt feature input into model, make output size to 128[/green underline]"
            )
            assert output_size >= 128, "output size must be greater than 128"
        if self.with_mask:
            logger.info("[green underline]with mask input[/green underline]")

        if only_y:
            logger.warning("output only Y channel of images")
            _load_type = ImageLoadType.GRAY8
        else:
            _load_type = ImageLoadType.RGB8
        load_keys = {"over": "OVER", "under": "UNDER"}

        if device is not None:
            self.device = th.device(device)
            self.cuda_stream = th.cuda.stream(th.cuda.Stream(device=self.device))
        else:
            self.device = "cpu"
            self.cuda_stream = nullcontext()

        if self.mode in ["train", "test"]:
            file_names = self.data_dir / f"{mode}.txt"
            assert file_names.exists(), "file names not found."
            file_names = np.loadtxt(file_names, dtype=str)
        elif self.mode == "all":
            logger.info(f"loading all data in dir {self.data_dir}")
            file_names = os.listdir(os.path.join(self.data_dir, load_keys["over"]))
            # hack the mode to be 'train'
            self.mode = "train"

        self.over_paths = [self.data_dir / load_keys["over"] / fn for fn in file_names]
        self.under_paths = [
            self.data_dir / load_keys["under"] / fn for fn in file_names
        ]
        assert len(self.over_paths) == len(
            self.under_paths
        ), "number of over/under exposure image pairs are not matched"
        if self.with_mask:
            self.mask_paths = [self.data_dir / "mask" / fn for fn in file_names]
            assert len(self.mask_paths) == len(
                self.over_paths
            ), "number of mask files are not matched"

        # check
        sort_keys = lambda x: x.stem
        self.over_paths.sort(key=sort_keys)
        self.under_paths.sort(key=sort_keys)
        if self.with_mask:
            self.mask_paths.sort(key=sort_keys)

        assert len(self.over_paths) == len(
            self.under_paths
        ), "number of files are not matched"

        # load images
        self.over_files, self.under_files = [], []
        for over_p, under_p in track(
            zip(self.over_paths, self.under_paths),
            description="loading images...",
            total=len(self.over_paths),
            disable=not is_main_process(),
        ):
            self.over_files.append(load_image(over_p, _load_type))
            self.under_files.append(load_image(under_p, _load_type))

        if self.with_mask:
            self.mask_files = []
            for mask_p in self.mask_paths:
                self.mask_files.append(load_image(mask_p, ImageLoadType.UNCHANGED)[0])

        logger.info("loaded all images done.")
        logger.info(
            "found {} over/under exposure image pairs for mode {}".format(
                len(self.over_files), mode
            )
        )

        # load txt file
        if self.with_txt:
            txt_file = self.data_dir / "t5_feature_MEF-MEFB.safetensors"
            logger.info("loading txt file: {}".format(txt_file))
            self.txt_feature = load_file(txt_file)
            logger.info("txt feature loaded")

        self.output_size = output_size

        # transforms
        self.img_resample = Resample.NEAREST
        self.img_align_corners = True if self.img_resample != Resample.NEAREST else None

        if transform is not None or self.mode == "test":
            self.default_transform = None
            self.aug_transform = None
        else:
            data_keys = ["input", "input"]
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
                        resample=self.img_resample,
                        ratio=(0.8, 1.4),
                        scale=_scale,
                        keepdim=True,
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
                        resample=self.img_resample,
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

    @staticmethod
    def random_crop_resize(
        *imgs,
        output_size,
        interpolation=InterpolationMode.BILINEAR,
        params: tuple | None = None,
    ):
        img_sz = imgs[0].shape[-2:]

        min_crop_size = output_size

        if params is None:
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
        # else:
        #     print(f'no augmentation at iter {self.n_iters}')

        data["over"], data["under"] = imgs[:2]
        _idx = 2
        if self.with_mask:
            data["mask"] = imgs[_idx]

        return data

    def __getitem__(self, index) -> dict[str, th.Tensor | str]:
        name = self.over_paths[index].stem
        with self.cuda_stream:
            over = self.over_files[index].to(self.device).type(th.float32) / 255.0
            under = self.under_files[index].to(self.device).type(th.float32) / 255.0
            if self.with_txt:
                txt_feature = (
                    self.txt_feature[name][0]
                    .to(self.device, non_blocking=True)
                    .type(th.float32)
                )
            if self.with_mask:
                mask = self.mask_files[index].to(self.device).type(th.float32)

        # color or other augmentations
        aug_imgs = [over, under]
        if self.with_mask:
            aug_imgs.append(mask)
        outp = self.process_imgs(*aug_imgs)

        # organize the outputs
        # compose the boundary gt image
        outp["gt"] = th.cat([outp["over"], outp["under"]], dim=0)
        if self.with_txt:
            outp["txt"] = txt_feature
        if not self.with_mask:
            outp["mask"] = False
        if self.get_name:
            over_file = self.over_paths[index]
            outp["name"] = over_file.name

        if self.mode == "train" and self.stop_aug_when_n_iters > 0:
            self.n_iters += 1

        return outp

    def __len__(self):
        return len(self.over_files)


if __name__ == "__main__":
    ds = MEFBDataset(
        "/Data3/cao/ZiHanCao/datasets/MEF-MEFB",
        mode="all",
        transform_ratio=1.0,
        output_size=256,
        with_txt=True,
        with_mask=True,
        stop_aug_when_n_iters=20_000,
    )
    dl = DataLoader(ds, batch_size=8, shuffle=True, num_workers=0)

    for i, data in enumerate(dl):
        over, under, mask, gt = data["over"], data["under"], data["mask"], data["gt"]
        print(
            data["over"].shape,
            data["under"].shape,
            data["gt"].shape,
            data["mask"].shape,
        )
