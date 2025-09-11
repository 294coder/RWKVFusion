"""
Author: Zihan Cao

Date: 2024/6/22
"""


# medical havard dataset

import os
from pathlib import Path
import pathlib
from typing import Literal, Sequence
from warnings import warn
import h5py
import safetensors
import safetensors.numpy
import safetensors.torch
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from PIL import Image
from torchvision.io import read_image, ImageReadMode
from tqdm import tqdm
import kornia as K
from kornia.constants import Resample
import kornia.augmentation as Ka
from copy import deepcopy


from typeguard import typechecked

from utils.misc import default


from utils import easy_logger

logger = easy_logger()


def to_pesudo_corlor():
    cmap = plt.cm.inferno

    def _inner(pet_img: np.ndarray):
        norm = mcolors.Normalize(vmin=pet_img.min(), vmax=pet_img.max())
        color_pet = cmap(norm(pet_img)).transpose([2, 0, 1])[:3]

        return color_pet

    return _inner


def to_two_tuple(a: int):
    return (a, a)


class MedHarvardDataset(Dataset):
    @typechecked
    def __init__(
        self,
        base_dir: str = None,
        file: "h5py.File | None" = None,
        file_dict: "dict | None" = None,
        mode: Literal["train", "test"] = None,
        crop_size: int = 72,
        data_source: Literal["whu", "uestc", "xmu"] = None,
        task: Literal["CT-MRI", "PET-MRI", "SPECT-MRI"] = "SPECT-MRI",
        color_spect: bool = True,
        transform_ratio: float = 0.0,
        device: "str | torch.device" = "cpu",
        get_name: bool = False,
        with_mask: bool = False,
        reduce_label: bool = False,
        with_txt_feature: bool = False,
        only_resize: "Sequence[int] | None" = None,
    ):
        source1, source2 = task.split("-")
        logger.info(
            f"start loading dataset: Medical Harvard Dataset ({source1} and {source2} sources)"
        )
        self.get_name = get_name
        self.task = task
        self.data_source = data_source
        self.task_check()
        self.with_mask = with_mask
        self.with_txt_feature = with_txt_feature
        self.reduce_label = reduce_label

        self.only_resize: "tuple[int, int] | None" = None
        if self.with_txt_feature and mode == "train":
            assert data_source == "xmu", "only xmu dataset support txt feature"
            self.only_resize = default(only_resize, (256, 256))
        if with_mask:
            assert data_source == "xmu", "only xmu dataset support mask"

        if file is not None or file_dict is not None:
            warn(
                "`file` and `file_dict` will be deprecated in the future, please use `base_dir` instead",
                DeprecationWarning,
            )

        if base_dir is not None:
            self.base_dir = Path(base_dir)
            self.mode = mode
            assert mode in ["train", "test"], 'mode should be either "train" or "test"'
            assert self.base_dir.exists(), "base_dir does not exist"

            self.data_source = data_source
            if data_source == "whu":
                if mode == "train":
                    data_path = "whu/train/Medical_Dataset.h5"
                    assert not get_name, "data from h5 file, do not have file path"
                    self.data = h5py.File(str(self.base_dir / data_path), "r")
                else:
                    data_path = "whu/test"  # are png files
                    self.mri_data_paths = list(
                        (self.base_dir / "whu" / "test" / "MR-T1").glob("*.png")
                    )
                    self.spect_data_paths = list(
                        (self.base_dir / "whu" / "test" / "PET-I").glob("*.png")
                    )

                    # sorted_key = lambda x: int(x.name.split('.')[0])
                    # self.mri_data_paths = sorted(self.mri_data_paths, key=sorted_key)
                    # self.spect_data_paths = sorted(self.spect_data_paths, key=sorted_key)

                    # read images
                    self.mri = [
                        read_image(str(p), mode=ImageReadMode.GRAY)[0] / 255
                        for p in self.mri_data_paths
                    ]
                    self.spect = [
                        read_image(str(p), mode=ImageReadMode.GRAY)[0] / 255
                        for p in self.spect_data_paths
                    ]
            elif data_source == "uestc":
                data_path = (
                    "uestc/training_data/train_Harvard(with_up)x4.h5"
                    if mode == "train"
                    else "uestc/test_data/test_Harvard(with_up)x4.h5"
                )
                self.data = h5py.File(str(self.base_dir / data_path), "r")

            elif data_source == "xmu":
                # load train/test split
                _load_txt_engine = lambda txt_path: np.loadtxt(
                    txt_path, dtype="<U15", delimiter=" "
                )
                task_path = Path(base_dir) / "xmu" / self.task
                txt_path = Path(task_path / f"{mode}.txt")
                assert task_path.exists(), f"{task_path} does not exist"
                assert txt_path.exists(), f'train test should be split first and saved as "train.txt" and "test.txt"'

                txt_arr = _load_txt_engine(txt_path)
                source_1_arr = txt_arr[:, 1]
                source_2_arr = txt_arr[:, 2]

                self.mri_data_paths = source_1_arr

                logger.info(f"found {len(source_1_arr)} images")

                # load txt features
                if with_txt_feature:
                    logger.info(f"loading txt features for {source1} and {source2}")
                    self.txt_features_s1 = safetensors.torch.load_file(
                        Path(task_path) / f"t5_feature_{task}_{source1}.safetensors"
                    )
                    self.txt_features_s2 = safetensors.torch.load_file(
                        Path(task_path) / f"t5_feature_{task}_{source2}.safetensors"
                    )

                # load images
                self.mri = []
                spect = []
                if with_mask:
                    mask_mri = []
                    mask_spect = []
                    logger.info(f"loading masks for {source1} and {source2}")

                for s1_p, s2_p in tqdm(
                    zip(source_1_arr, source_2_arr), leave=True, total=len(source_1_arr)
                ):
                    self.mri.append(np.array(Image.open(os.path.join(task_path, s1_p))))
                    spect.append(np.array(Image.open(os.path.join(task_path, s2_p))))
                    if with_mask:
                        mask_mri.append(
                            np.array(
                                Image.open(
                                    os.path.join(task_path, "masks", s1_p)
                                ).convert("L")
                            )
                        )
                        mask_spect.append(
                            np.array(
                                Image.open(
                                    os.path.join(task_path, "masks", s2_p)
                                ).convert("L")
                            )
                        )

                # to tensor
                if with_mask:
                    self.mask_mri = (
                        torch.from_numpy(np.stack(mask_mri, axis=0))
                        .unsqueeze(1)
                        .type(torch.float32)
                    )
                    self.mask_spect = (
                        torch.from_numpy(np.stack(mask_spect, axis=0))
                        .unsqueeze(1)
                        .type(torch.float32)
                    )

                    if reduce_label:
                        self.mask_mri[self.mask_mri > 0] = 1
                        self.mask_spect[self.mask_spect > 0] = 1

                self.mri = np.stack(self.mri, axis=0)[:, np.newaxis] / 255.0
                if self.task == "CT-MRI":
                    spect = np.stack(spect, axis=0)[:, np.newaxis] / 255.0
                else:
                    spect = np.stack(spect, axis=0).transpose([0, 3, 1, 2]) / 255.0

                if self.only_resize is not None:
                    resizer = Ka.Resize(align_corners=True, size=self.only_resize)
                    self.mri = resizer(torch.from_numpy(self.mri).type(torch.float32))
                    spect = resizer(torch.from_numpy(spect).type(torch.float32))
                    if self.with_mask:
                        mask_resizer = Ka.Resize(
                            align_corners=True,
                            size=self.only_resize,
                            resample="nearest",
                        )
                        self.mask_mri = mask_resizer(self.mask_mri)
                        self.mask_spect = mask_resizer(self.mask_spect)
                else:
                    # basic augmentaions
                    # TODO: add mask augmentation
                    self.augs = [
                        Ka.RandomResizedCrop(
                            to_two_tuple(crop_size), p=1, keepdim=True
                        ),
                        Ka.RandomVerticalFlip(p=transform_ratio, keepdim=True),
                        Ka.RandomHorizontalFlip(p=transform_ratio, keepdim=True),
                        Ka.RandomRotation((-0.2, 0.2), p=transform_ratio, keepdim=True),
                    ]
                self.additional_augs = [
                    Ka.ColorJiggle(
                        brightness=0.0,
                        contrast=0.1,
                        saturation=0.1,
                        hue=0.1,
                        p=transform_ratio,
                        keepdim=True,
                    ),
                    Ka.RandomErasing(
                        scale=(0.02, 0.2),
                        ratio=(0.3, 3.3),
                        p=transform_ratio,
                        keepdim=True,
                    ),
                ]

        elif file is not None:
            assert isinstance(
                file, h5py.File
            ), "file should be an instance of h5py.File"
            assert data_source is not None, "data_source should be provided"
            self.data = file
            self.data_source = data_source

        elif file_dict is not None:
            assert isinstance(
                file_dict, dict
            ), "file_dict should be an instance of dict"
            assert data_source is not None, "data_source should be provided"

            if data_source == "uestc":
                assert "MRI" in file_dict.keys(), 'file_dict should have key "MRI"'
                assert (
                    "SPECT_y" in file_dict.keys()
                ), 'file_dict should have key "SPECT_y"'
            else:
                assert "data" in file_dict.keys(), 'file_dict should have key "data"'

            self.data = file_dict
            self.data_source = data_source

        else:
            raise ValueError(
                "either `base_dir` or `file` or `file_dict` should be provided"
            )

        if hasattr(self, "data"):
            if (
                "data" in list(self.data.keys()) or self.data_source == "whu"
            ):  # is whu harvard dataset
                self.mri = self.data["data"][:, 0:1]
                spect = self.data["data"][:, 1:2]
            else:  # is re-clipped dataset from xiao-woo
                self.mri = self.data["MRI"][:].transpose([2, 0, 1])
                spect = self.data["SPECT_y"][:].transpose([2, 0, 1])
        elif mode == "test" and data_source == "whu":  # for mode='test' in whu dataset
            self.mri = np.stack(self.mri, axis=0)[:, None]
            spect = np.stack(self.spect, axis=0)

        self.color_spect = color_spect
        if color_spect and data_source != "xmu":
            self.pet_recolored_fn = to_pesudo_corlor()
            colored_shape = (spect.shape[0], 3, *(spect.shape[-2:]))
            self.recolored_spect: np.ndarray = np.zeros(colored_shape, dtype=np.float32)
            self.to_color(spect)  # set `self.recolored_spect` inside the function
        else:
            self.recolored_spect = spect

        # to tensor
        if not torch.is_tensor(self.mri):
            self.mri = torch.from_numpy(self.mri).contiguous().type(torch.float32)
            self.recolored_spect = (
                torch.from_numpy(self.recolored_spect).contiguous().type(torch.float32)
            )

        # assert size equal of two modalities
        if self.mri.shape[-2:] != self.recolored_spect.shape[-2:]:  # size not equal
            logger.info(
                f"MRI and SPECT images are not in the same size {tuple(self.mri.shape[-2:])} | "
                f"{tuple(self.recolored_spect.shape[-2:])}, resizing..."
            )
            self.recolored_spect = torch.nn.functional.interpolate(
                self.recolored_spect, size=self.mri.shape[-2:], mode="bilinear"
            )

        self.device = device
        self.on_cuda = "cuda" in str(self.device)
        self.device_context = (
            torch.cuda.Stream(device=self.device) if self.on_cuda else None
        )

        # print MRI and SPECT data shape
        logger.info("{:^30} {:^30}".format(source1, source2))
        logger.info(
            "{:^30} {:^30}".format(
                str(tuple(self.mri.size())), str(tuple(self.recolored_spect.size()))
            )
        )

    def _get_kornia_mask_flags(
        selg,
        aug,
    ):
        _mask_flags = deepcopy(aug.flags)
        if "resample_method" in _mask_flags:
            _mask_flags["resample_method"] = Resample.get("nearest")
        if "resample" in _mask_flags:
            _mask_flags["resample"] = Resample.get("nearest")

        return _mask_flags

    def apply_transform(
        self, s1: torch.Tensor, s2: torch.Tensor, mask: torch.Tensor | None = None
    ):
        if hasattr(self, "augs") and self.mode == "train":
            for aug in self.augs:
                aug: "K.augmentation.AugmentationBase2D"

                s1 = aug(s1)
                s2 = aug(s2, params=aug._params)

                if mask is not None:
                    mask_flags = self._get_kornia_mask_flags(aug)
                    mask = aug(mask, params=aug._params, **mask_flags)

        if hasattr(self, "additional_augs") and self.mode == "train":
            for aug in self.additional_augs:
                aug: "K.augmentation.IntensityAugmentationBase2D"

                if isinstance(aug, Ka.ColorJiggle):
                    s2 = aug(s2)
                else:
                    s1 = aug(s1)
                    s2 = aug(s2, params=aug._params)

                    if mask is not None:
                        mask_flags = self._get_kornia_mask_flags(aug)
                        mask = aug(mask, params=aug._params, **mask_flags)

        return s1, s2, mask

    def to_color(self, pet_images: np.ndarray):
        for i in range(pet_images.shape[0]):
            if pet_images[i].shape[0] == 1:  # [1, h, w]
                pet_img = pet_images[i][0]
            else:
                pet_img = pet_images[i]
            self.recolored_spect[i] = self.pet_recolored_fn(pet_img)

    def task_check(self):
        if self.data_source in ["whu", "uestc"]:
            if self.task != "MRI-SPECT":
                logger.warning(
                    f"task of dataset source {self.data_source} only supports MRI-SPECT"
                )
            self.task = "MRI-SPECT"

    def __len__(self):
        return len(self.recolored_spect)

    def __getitem__(self, idx):
        mri, spect = self.mri[idx], self.recolored_spect[idx]

        # prepare output as a dict
        outp = {}
        name = os.path.basename(self.mri_data_paths[idx])
        if self.get_name:
            outp["name"] = name

        if self.with_txt_feature:
            s1_name = name.split(".")[0]
            s1_txt_feature = self.txt_features_s1[s1_name][0]
            s2_txt_feature = self.txt_features_s2[s1_name][0]
            txt = torch.cat([s1_txt_feature, s2_txt_feature], dim=-1)

        if self.with_mask:
            mask_mri = self.mask_mri[idx]
            mask_spect = self.mask_spect[idx]
            mask = torch.cat([mask_spect, mask_mri], dim=0)
        else:
            mask = False

        mri, spect, mask = self.apply_transform(mri, spect, mask)
        gt = torch.cat([spect, mri], dim=0)

        if self.on_cuda:
            with torch.cuda.stream(self.device_context):
                outp["s1"] = spect.to(self.device, non_blocking=True)
                outp["s2"] = mri.to(self.device, non_blocking=True)
                outp["gt"] = gt.to(self.device, non_blocking=True)
                if self.with_mask:
                    outp["mask"] = mask.to(self.device, non_blocking=True)
                if self.with_txt_feature:
                    outp["txt"] = txt.to(self.device, non_blocking=True)
        else:
            outp["s1"] = spect
            outp["s2"] = mri
            outp["gt"] = gt
            if self.with_txt_feature:
                outp["txt"] = txt

        return outp


if __name__ == "__main__":
    from PIL import Image

    ds = MedHarvardDataset(
        base_dir="/Data3/cao/ZiHanCao/datasets/MedHarvard",
        mode="train",
        data_source="xmu",
        device="cuda:0",
        transform_ratio=1.0,
        task="SPECT-MRI",
        get_name=True,
        with_mask=True,
        reduce_label=True,
        with_txt_feature=True,
        only_resize=(256, 256),
    )

    dl = DataLoader(ds, batch_size=16, shuffle=False)
    # print(ds[0])

    # saved_dir = Path('/Data3/cao/ZiHanCao/datasets/MedHarvard/whu/test/PET_recolored')
    for i, data in enumerate(dl, 1):
        spect, mri, mask, gt, name = (
            data["s1"],
            data["s2"],
            data["mask"],
            data["gt"],
            data["name"],
        )
        # print(mri.shape, spect.shape, mask.shape, gt.shape)

        pass

        # plot images
        # fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        # axes[0].imshow(mri[0, 0].cpu().numpy(), cmap='gray')
        # axes[0].set_title('MRI')
        # axes[1].imshow(spect[0].permute(1, 2, 0).cpu().numpy())
        # axes[1].set_title('SPECT')

        # plt.savefig(f'test_imgs/test_{i}.png')

        # if i > 10:
        #     break

    # for i, data in enumerate(dl, 1):
    #     pass

    # for i, data in enumerate(dl, 1):
    #     pass

    # for i, data in enumerate(dl, 1):
    #     pass
