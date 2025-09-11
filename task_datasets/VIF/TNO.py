import functools
import os
from copy import deepcopy
from glob import glob
from pathlib import Path
from typing import Literal, Sequence, Union

import kornia.augmentation as K
import PIL.Image as Image
import torch
import torch.utils.data as Data
import torchvision.transforms as transforms
from beartype import beartype
from kornia.constants import Resample
from safetensors.numpy import load_file
from typeguard import typechecked

from utils import default, easy_logger

logger = easy_logger(func_name="TNO Dataset")


class TNODataset(Data.Dataset):
    """
    Infrared dataset

    inter output:
        vis, ms_vis, ir, gt(cat[ir, vis])
    """

    @beartype
    def __init__(
        self,
        base_dir: str,
        mode: Literal["train", "test"],
        size: int | Sequence[int] | None = 72,
        no_split=False,  # when set to True, the dataloader batch size should be set to 1
        aug_prob=0.0,
        duplicate_vis_channel: bool = True,
        get_downsampled_vis: bool = False,
        fast_eval_n_samples: int | None = None,
        get_name: bool = False,
        device: Union[str, torch.device] = None,
        with_txt_feature: bool = False,
        only_resize: Sequence[int] | None = None,
        output_none_mask: bool = True,
    ):
        self.mode = mode
        self.base_dir = base_dir
        self.no_split = no_split
        self.size = size
        self.get_downsampled_vis = get_downsampled_vis
        self.get_name = get_name
        self.fast_eval_n_samples = fast_eval_n_samples if mode == "test" else None
        self.mode = mode
        self.device = device
        self.with_txt_feature = with_txt_feature
        self.only_resize = only_resize
        self.output_none_mask = output_none_mask

        if not no_split and mode == "train":
            logger.debug(
                "TNO dataset has not the same size of images, the batch size should be set to 1."
            )

        if mode == "test":
            self.no_split = True

        if self.device is not None:
            self.cuda_stream = torch.cuda.stream(torch.cuda.Stream(device))

        if mode == "train":
            mode = "new_training_data"
            # mode = 'new_training_data'
            # print('TNO dataset, using new training data')
            infrared_name = "ir"
            vis_name = "vi"
        else:  # test
            mode = "new_test_data"
            # mode = 'only_RS_testset'
            infrared_name = "ir"
            vis_name = "vi"

        self.infrared_paths = sorted(
            glob(base_dir + f"/{mode}/{infrared_name}/*"), key=os.path.basename
        )
        self.vis_paths = sorted(
            glob(base_dir + f"/{mode}/{vis_name}/*"), key=os.path.basename
        )

        if fast_eval_n_samples and mode == "test":
            self.infrared_paths = self.infrared_paths[:fast_eval_n_samples]
            self.vis_paths = self.vis_paths[:fast_eval_n_samples]

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

            assert (
                len(self.txt_feature_vi)
                == len(self.txt_feature_ir)
                == len(self.infrared_paths)
            ), "the length of t5 features should be the same as the number of images"

        logger.info(
            f"{mode} file - num of files: {len(self.infrared_paths)}, size {size if not no_split else 'no split'} "
        )

        to_tensor = transforms.ToTensor()

        read_vis_mode = "RGB" if duplicate_vis_channel else "L"

        if duplicate_vis_channel:
            logger.debug("dataset will output 3 channel vis image")
            logger.debug("output will be (vis, ir, gt)")
        else:
            logger.debug("output will (ir, downsampled(vis), vis, gt)")

        self.ir_imgs = []
        self.vis_imgs = []
        for ir_p, vi_p in zip(self.infrared_paths, self.vis_paths):
            ir_img = Image.open(ir_p)
            vi_img = Image.open(vi_p)

            ir_img = self.check_convert(ir_img)
            vi_img = self.check_convert(vi_img, mode=read_vis_mode)

            # print(ir_img.size, vi_img.size)
            if ir_img.size != vi_img.size:
                logger.warning(
                    os.path.basename(ir_p),
                    os.path.basename(vi_p),
                    ir_img.size,
                    vi_img.size,
                )
                minimum_size = (
                    vi_img.size if vi_img.size[0] < ir_img.size[0] else vi_img.size
                )

                ir_img = ir_img.resize(minimum_size, resample=Image.Resampling.NEAREST)
            self.ir_imgs.append(to_tensor(ir_img))
            self.vis_imgs.append(to_tensor(vi_img))

        # print info
        if self.mode == "train":
            _shape = list(self.vis_imgs[0].shape)
            logger.info("{:^20} {:^20}".format("vis", "ir"))
            logger.info(
                "{:^20} {:^20}".format(
                    str((len(self.vis_imgs), *_shape)),
                    str((len(self.ir_imgs), *_shape)),
                )
            )

        self.gt = [
            torch.cat([vis, ir], dim=0) for vis, ir in zip(self.vis_imgs, self.ir_imgs)
        ]

        self.down_sample = functools.partial(
            torch.nn.functional.interpolate,
            scale_factor=1 / 4,
            mode="nearest",
        )

        resample_type = Resample.BILINEAR
        if only_resize is None and not with_txt_feature:
            self.crop_fn = K.AugmentationSequential(
                K.RandomResizedCrop(
                    size=(size, size),
                    scale=(0.8, 1.0),
                    keepdim=True,
                    resample=resample_type,
                ),
                data_keys=["input", "input", "input"],
                same_on_batch=True,
            )
            if aug_prob > 0:
                self.random_aug_K = K.AugmentationSequential(
                    K.RandomVerticalFlip(p=aug_prob, keepdim=True),
                    K.RandomHorizontalFlip(p=aug_prob, keepdim=True),
                    K.RandomSharpness(p=aug_prob, keepdim=True),
                    K.RandomRotation(
                        degrees=(-20, 20),
                        p=aug_prob,
                        keepdim=True,
                        resample=resample_type,
                    ),
                    data_keys=["input", "input", "input"],
                    same_on_batch=True,
                )
            self.random_crop_ms = transforms.RandomCrop(size // 4)
        else:
            self.only_resize = default(only_resize, (224, 280))
            self.crop_fn = K.AugmentationSequential(
                transforms.Resize(self.only_resize),
                data_keys=["input", "input", "input"],
                same_on_batch=True,
            )
            if aug_prob > 0:
                self.random_aug_K = K.AugmentationSequential(
                    K.RandomVerticalFlip(p=aug_prob, keepdim=True),
                    K.RandomHorizontalFlip(p=aug_prob, keepdim=True),
                    K.RandomSharpness(p=aug_prob, keepdim=True),
                    K.RandomRotation(
                        degrees=(-20, 20),
                        p=aug_prob,
                        keepdim=True,
                        resample=resample_type,
                    ),
                    data_keys=["input", "input", "input"],
                    same_on_batch=True,
                )

    def check_convert(self, x: Image.Image, mode: str | None = None):
        if mode is not None:
            x = x.convert(mode)
        else:
            if len(x.mode) > 2:
                x, *_ = x.convert("YCbCr").split()
                # print('debug: convert ycbcr')
            else:  # only gray
                x = x.convert("L")  # this is not needed, but keep anyway
        return x

    def __len__(self):
        return len(self.ir_imgs)

    def process_data(self, *imgs):
        imgs = self.crop_fn(*imgs)
        if hasattr(self, "random_aug_K") and self.mode == "train":
            imgs = self.random_aug_K(*imgs)
        return imgs

    def __getitem__(self, index):
        ir = self.ir_imgs[index]
        vis = self.vis_imgs[index]
        gt = self.gt[index]

        if self.with_txt_feature:
            vi_name = os.path.basename(self.vis_paths[index]).split(".")[0]
            ir_name = os.path.basename(self.infrared_paths[index]).split(".")[0]
            ir_txt = torch.from_numpy(self.txt_feature_ir[ir_name][0]).to(torch.float32)
            vi_txt = torch.from_numpy(self.txt_feature_vi[vi_name][0]).to(torch.float32)

        if self.device is not None:
            with self.cuda_stream:
                ir = ir.to(self.device)
                vis = vis.to(self.device)
                gt = gt.to(self.device)
                if self.with_txt_feature:
                    vi_txt = vi_txt.to(self.device)
                    ir_txt = ir_txt.to(self.device)

        # put tensors to device and process with augmentations
        if not self.no_split and self.mode == "train":
            vis, ir, gt = self.process_data(vis, ir, gt)

        # organize output
        if self.get_downsampled_vis:  # output v1
            output = {
                "ir": ir,
                "downsampled_vis": self.down_sample(vis[None])[0],
                "vis": vis,
                "gt": gt,
            }
        else:  # output v2
            output = {"vi": vis, "ir": ir, "gt": gt}

        if self.output_none_mask:
            output["mask"] = False

        if self.get_name:
            name = os.path.basename(self.infrared_paths[index])
            output["name"] = name

        if self.with_txt_feature:
            output["txt"] = torch.cat([vi_txt, ir_txt], dim=-1)

        return output

    def __iter__(self):
        return self


if __name__ == "__main__":
    import matplotlib.gridspec as gridspec
    import matplotlib.pyplot as plt

    dataset = TNODataset(
        "/Data3/cao/ZiHanCao/datasets/RoadScene_and_TNO",
        mode="train",
        no_split=False,
        aug_prob=1.0,
        get_name=True,
        duplicate_vis_channel=True,
        with_txt_feature=True,
        only_resize=None,
        output_none_mask=True,
    )
    dl = Data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    for i, data in enumerate(dl):
        vis, ir, mask, gt, path = (
            data["vi"],
            data["ir"],
            data["mask"],
            data["gt"],
            data.get("name"),
        )

        print(vis.shape, ir.shape, mask.shape, gt.shape, path)

        # fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        # axes = axes.flatten()

        # print(ir.shape)

        # inv_ir = img_padder.inverse(ir_pad)

        # axes[0].imshow(ir[0].permute(1, 2, 0), "gray")
        # axes[1].imshow(ir_pad[0].permute(1, 2, 0), "gray")
        # axes[2].imshow(inv_ir[0].permute(1, 2, 0), "gray")

        # fig.savefig(f'./pad_{i}.png', dpi=200, bbox_inches='tight')

    # grid = gridspec.GridSpec(2, 2)
    # axes = [
    #     plt.subplot(grid[0, 0]),
    #     plt.subplot(grid[0, 1]),
    #     plt.subplot(grid[1, 0]),
    #     plt.subplot(grid[1, 1]),
    # ]
    # axes[0].imshow(ir[0].permute(1, 2, 0), "gray")
    # axes[0].set_title("ir")
    # axes[1].imshow(ms[0].permute(1, 2, 0), "gray")
    # axes[1].set_title("ms_vis")
    # axes[2].imshow(vis[0].permute(1, 2, 0), "gray")
    # axes[2].set_title("vis")
    # # only show channel 3 image
    # axes[3].imshow(torch.cat([gt[0], gt[0, 0:1]], dim=0).permute(1, 2, 0))
    # axes[3].set_title("gt")
    # plt.show()
    # # fig = plt.gcf()
    # # fig.savefig(f'./{i}.png')
    #
    # if i > 4:
    #     break
