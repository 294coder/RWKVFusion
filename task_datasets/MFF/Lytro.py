"""
Lytro dataset
used as test dataset

Author: Zihan Cao
Date: 2024.07.16

UESTC copyright(c)
"""

from contextlib import nullcontext
from pathlib import Path
from typing import Literal

import numpy as np
import torch as th
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from safetensors.torch import load_file
from kornia.io import load_image, ImageLoadType
from rich.progress import track

from utils import easy_logger

logger = easy_logger(func_name="LytroDataset")


class LytroDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        get_name: bool = True,
        with_mask: bool = False,
        with_txt: bool = False,
    ):
        self.data_dir = Path(data_dir)
        self.with_mask = with_mask
        self.with_txt = with_txt
        self.get_name = get_name

        self.far_path = list((self.data_dir / "FAR").glob("*.*"))
        self.near_path = list((self.data_dir / "NEAR").glob("*.*"))
        if with_mask:
            self.mask_path = list((self.data_dir / "mask").glob("*.*"))

        sort_key = lambda x: int(x.stem)
        self.far_path.sort(key=sort_key)
        self.near_path.sort(key=sort_key)
        if with_mask:
            self.mask_path.sort(key=sort_key)

        self.far_images = []
        self.near_image = []
        self.mask_images = []
        for f, n in track(zip(self.far_path, self.near_path), total=len(self.far_path)):
            assert f.stem == n.stem, "far and near file names are not matched."

            self.far_images.append(load_image(f, ImageLoadType.RGB8))
            self.near_image.append(load_image(n, ImageLoadType.RGB8))

        if with_mask:
            self.masks = []
            for m in self.mask_path:
                self.mask_images.append(load_image(m, ImageLoadType.UNCHANGED)[0])

        if with_txt:
            logger.info(
                f'load t5 feature from {self.data_dir / "t5_feature_lytro.safetensors"}'
            )
            self.txt_feature = load_file(self.data_dir / "t5_feature_lytro.safetensors")

    def __len__(self):
        return len(self.far_path)

    def __getitem__(self, index):
        far, near = (
            self.far_images[index] / 255.0,
            self.near_image[index] / 255.0,
        )

        outp = {"far": far, "near": near, "gt": th.cat([far, near], dim=0)}
        if self.get_name:
            name = self.far_path[index].name
            outp["name"] = name
        if self.with_mask:
            mask = self.mask_images[index]
            outp["mask"] = mask
        if self.with_txt:
            name = self.far_path[index].stem
            txt_feature = self.txt_feature[name][0].type(th.float32)
            outp["txt"] = txt_feature

        return outp


if __name__ == "__main__":
    dataset = LytroDataset(
        "/Data3/cao/ZiHanCao/datasets/MFF-Lytro", with_txt=True, with_mask=True
    )
    dl = DataLoader(dataset, batch_size=12, shuffle=False)

    for data in dl:
        print(
            data["far"].shape,
            data["near"].shape,
            data["name"],
            data["mask"].shape,
            data["txt"].shape,
        )
        pass
