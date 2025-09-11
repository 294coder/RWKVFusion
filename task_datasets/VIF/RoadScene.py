import functools
from typing import Literal, Union

import torch
import torch.utils.data as Data
import os
import numpy as np
import torchvision.transforms as transforms
from glob import glob
import PIL.Image as Image
from pathlib import Path
from safetensors.torch import load_file
from loguru import logger

from utils import default


class RoadSceneDataset(Data.Dataset):
    """
    Infrared dataset

    inter output:
        vis, ms_vis, ir, gt(cat[ir, vis])
    """

    def __init__(
        self,
        base_dir: str,
        mode: Literal["train", "test"],
        size: int = 128,
        no_split: bool = False,
        duplicate_vis_channel: bool = False,
        get_downsampled_vis: bool = False,
        get_name: bool = False,
        with_txt_feature: bool = False,
        only_resize: "tuple[int, int] | None" = None,
        output_none_mask: bool = True,
    ):
        assert mode in ["train", "validation", "test"]
        self.mode = mode
        self.base_dir = base_dir
        self.no_split = no_split
        self.duplicate_vis_channel = duplicate_vis_channel
        self.get_downsampled_vis = get_downsampled_vis
        self.get_name = get_name
        self.with_txt_feature = with_txt_feature
        self.only_resize = only_resize
        self.output_none_mask = output_none_mask

        if duplicate_vis_channel:
            logger.debug("dataset will output 3 channel vis image")
            logger.debug("output will be [vis, ir, gt]")

        if get_downsampled_vis:
            logger.debug("output will [ir, downsampled(vis), vis, gt]")

        if mode == "train":
            infrared_name = "infrared"
            vis_name = "visible"
            suffix = "jpg"
        else:
            infrared_name = "ir test"
            vis_name = "vi test"
            suffix = "bmp"

        self.infrared_paths = glob(base_dir + f"/{mode}/{infrared_name}/*.{suffix}")
        self.vis_paths = glob(base_dir + f"/{mode}/{vis_name}/*.{suffix}")

        if mode == "test":
            key = lambda x: int(os.path.basename(x.strip("." + suffix)))
            self.vis_paths.sort(key=key)
            self.infrared_paths.sort(key=key)

        to_tensor = transforms.ToTensor()

        # load t5 features
        if with_txt_feature:
            t5_feature_ps = list((Path(base_dir) / mode).glob("*.safetensors"))
            assert (
                len(t5_feature_ps) == 2
            ), "we only support two t5 features for vi and ir"

            logger.info(f"load t5 features from {t5_feature_ps}")
            for t5_feature_p in t5_feature_ps:
                if "vi" in t5_feature_p.as_posix():
                    self.txt_feature_vi = load_file(t5_feature_p)
                elif "ir" in t5_feature_p.as_posix():
                    self.txt_feature_ir = load_file(t5_feature_p)
                else:
                    raise ValueError(
                        f"not found t5 feature for vi or ir in {t5_feature_p}"
                    )

        vi_read_mode = "L" if duplicate_vis_channel else "RGB"
        self.ir_imgs = [
            to_tensor(Image.open(path).convert("L")) for path in self.infrared_paths
        ]
        self.vis_imgs = [
            to_tensor(Image.open(path).convert(vi_read_mode)) for path in self.vis_paths
        ]

        self.gt = [
            torch.cat([vis, ir], dim=0) for ir, vis in zip(self.ir_imgs, self.vis_imgs)
        ]

        self.random_crop_ori = transforms.RandomCrop(size)
        self.down_sample = functools.partial(
            torch.nn.functional.interpolate, scale_factor=1 / 4, mode="bilinear"
        )
        self.random_crop_ms = transforms.RandomCrop(size // 4)

        if with_txt_feature:
            self.only_resize = default(only_resize, (224, 280))
            self.resize_fn = transforms.Resize(self.only_resize)

    def __len__(self):
        return len(self.ir_imgs)

    def process_data(self, *imgs):
        processed_imgs = []
        rng_state = torch.get_rng_state()
        for img in imgs:
            torch.set_rng_state(rng_state)
            img = self.random_crop_ori(img)
            processed_imgs.append(img)
        return processed_imgs

    def __getitem__(self, index):
        ir = self.ir_imgs[index]
        vis = self.vis_imgs[index]
        gt = self.gt[index]
        if not self.no_split or self.mode == "train":
            vis, ir, gt = self.process_data(vis, ir, gt)

        output = {"vi": vis, "ir": ir, "gt": gt}

        if self.get_downsampled_vis:
            output["downsampled_vis"] = self.down_sample(vis[None])[0]

        if self.output_none_mask:
            output["mask"] = False

        if self.get_name:
            name = os.path.basename(self.vis_paths[index])
            output["name"] = name

        if self.with_txt_feature:
            vi_name = Path(self.vis_paths[index]).stem
            ir_name = Path(self.infrared_paths[index]).stem
            vi_txt_feat = self.txt_feature_vi[vi_name][0].type(torch.float32)
            ir_txt_feat = self.txt_feature_ir[ir_name][0].type(torch.float32)
            output["txt"] = torch.cat([vi_txt_feat, ir_txt_feat], dim=-1)

        if self.only_resize is not None and self.mode == "train":
            output["vi"] = self.resize_fn(output["vi"])
            output["ir"] = self.resize_fn(output["ir"])
            output["gt"] = self.resize_fn(output["gt"])

        return output


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    dataset = RoadSceneDataset(
        r"/Data3/cao/ZiHanCao/datasets/RoadSceneFusion",
        "test",
        no_split=True,
        duplicate_vis_channel=False,
        with_txt_feature=True,
        only_resize=(224, 280),
        get_name=True,
    )
    dl = Data.DataLoader(dataset, batch_size=1, shuffle=True)
    for i, data in enumerate(dl):
        vis, ir, gt, path = data["vi"], data["ir"], data["gt"], data.get("name")
        # print(ir.max(), vis.max(), gt.max())

        # shape
        print(vis.shape, ir.shape, gt.shape, data["txt"].shape)

        # grid = gridspec.GridSpec(2, 2)
        # axes = [
        #     plt.subplot(grid[0, 0]),
        #     plt.subplot(grid[0, 1]),
        #     plt.subplot(grid[1, 0]),
        #     plt.subplot(grid[1, 1]),
        # ]
        # axes[0].imshow(vis[0].permute(1, 2, 0), "gray")
        # axes[0].set_title("vis")
        # # axes[1].imshow(ms[0].permute(1, 2, 0), "gray")
        # # axes[1].set_title("ms_vis")
        # axes[2].imshow(ir[0].permute(1, 2, 0), "gray")
        # axes[2].set_title("ir")
        # # only show channel 3 image
        # axes[3].set_title("gt")
        # axes[3].imshow(torch.cat([gt[0], gt[0, 0:1]], dim=0).permute(1, 2, 0))
        # plt.show()

        # plt.savefig(f'RS_test_{i}.png')
        # plt.clf()

        # if i > 10:
        #     break
