from pathlib import Path
from typing import Literal

import torch
from beartype import beartype
from kornia.io import ImageLoadType, load_image
from torch.utils.data import DataLoader, Dataset

# from beartype.door import is_bearable
from utils import (
    default,
    easy_logger,
)

logger = easy_logger(func_name="paired_dataset")


ALL_TASKS = [
    # 'Pansharpening',
    # 'HMIF',
    "VIF",
    "MEF",
    "MFF",
    "medical_fusion",
]
DATASET_KEYS = {
    "pansharpening": ["ms", "lms", "pan", "gt", "txt"],
    "HMIF": ["rgb", "lr_hsi", "hsi_up", "gt", "txt"],
    "VIF": ["vi", "ir", "mask", "gt", "txt", "name"],
    "MEF": ["over", "under", "gt", "txt", "name"],
    "MFF": ["far", "near", "gt", "txt", "name"],
    "medical_fusion": ["s1", "s2", "mask", "gt", "txt", "name"],
}


class PairedDataset(Dataset):
    @beartype
    def __init__(
        self,
        dir1: str,
        dir2: str,
        task_type: Literal["VIF", "MEF", "MFF", "medical_fusion"],
        img_keys: list[str] | None = None,
    ):
        super().__init__()
        self.task_type = task_type
        self.ret_img_keys = default(img_keys, DATASET_KEYS[task_type][:2])
        logger.info(
            "paired dataset can be only used for image pairs to fuse paired images"
        )
        logger.info("returning image keys:", self.ret_img_keys)

        self.dir1 = Path(dir1)
        self.dir2 = Path(dir2)

        # glob imgs
        _POSSIBLE_EXTENSIONS = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff"]
        self.imgs1_path = []
        self.imgs2_path = []
        for ext in _POSSIBLE_EXTENSIONS:
            self.imgs1_path.extend(self.dir1.glob(ext))
            self.imgs2_path.extend(self.dir2.glob(ext))

        # sort
        sorted_key = lambda x: x.stem
        self.imgs1_path = sorted(self.imgs1_path, key=sorted_key)
        self.imgs2_path = sorted(self.imgs2_path, key=sorted_key)

        # asserts
        assert (
            len(self.imgs1_path) == len(self.imgs2_path)
        ), f"number of images in both directories should be same, but got {len(self.imgs1_path)} and {len(self.imgs2_path)}"

    def __len__(self):
        return len(self.imgs1_path)

    def __getitem__(self, idx) -> dict[str, torch.Tensor | str]:
        img1 = load_image(str(self.imgs1_path[idx]), ImageLoadType.UNCHANGED)
        img2 = load_image(str(self.imgs2_path[idx]), ImageLoadType.UNCHANGED)
        _imgs = [img1, img2]

        # construct dict
        ret_imgs = {key: img for key, img in zip(self.ret_img_keys, _imgs)}
        ret_imgs["name"] = self.imgs1_path[idx].name

        return ret_imgs
