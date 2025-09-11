"""
                    LOADER USAGE GUIDANCE

1. Download the VIS-IR dataset: LLVIP, MSRS, M3FD.
2. Use the provided mask files (or grounding the masks using SAM predictor)
3. Load all datasets as you want and train your model 😄


Author: Zihan Cao
Date: 2024.06.27

UESTC, Copyright (c) 2024
"""

import copy
from random import shuffle
from typing import TYPE_CHECKING, Literal, Sequence, Union

import numpy as np
import torch
from nvidia.dali.plugin.base_iterator import LastBatchPolicy
from nvidia.dali.plugin.pytorch import DALIGenericIterator
from torch.utils.data import DataLoader, DistributedSampler

from task_datasets import (
    LLVIPDALIPipeLoader,
    M3FDDALIPipeLoader,
    MSRSDALIPipeLoader,
    TNODataset,
)
from utils import easy_logger

logger = easy_logger(func_name="VIS_IR_joint_pipe", level="INFO")

## helper functions


def to_two_tuple(x: "int | float"):
    return (x, x)


# to ascii code
def string_to_ascii_array(s):
    ascii_values = [ord(char) for char in s]
    return np.array(ascii_values, dtype=np.uint8)


def ascii_tensor_to_string(ascii_tensor):
    ascii_array = ascii_tensor.detach().cpu().numpy()

    string_s = []
    # for arr in ascii_array:
    characters = [chr(code) for code in ascii_array]
    string_s.append("".join(characters))

    return string_s


if TYPE_CHECKING:
    allowd_base_dirs = Literal["llvip", "msrs", "m3fd", "roadscene_tno_joint"]


class VISIRJointGenericLoader(object):
    def __init__(
        self,
        base_dirs: "dict[allowd_base_dirs, str]",  # e.g., {'llvip': 'path/to/LLVIP', 'msrs': 'path/to/MSRS', 'm3fd': 'path/to/M3FD'}
        mode: Literal[
            "train", "test"
        ],  # ensure the datasets has `train` and `test` files
        output_size: int | Sequence[int] | None = 72,  # output size of the image
        with_mask: bool = True,  # each batch will have a mask tensor
        with_txt_feature: bool = False,  # whether to use the txt feature
        batch_size: int = 32,  # batch size of training
        shard_id: int = 0,  # ddp process id
        num_shards: int = 1,  # ddp process number (world_size)
        n_thread: int = 4,  # number of threads for data loading
        device: "str | torch.device" = "cuda",  # device for one process loading
        random_datasets_getitem: bool = False,  # getitem will randomly select a dataset
        shuffle_in_dataset: bool = True,  # a selected dataset will be shuffled
        *,
        fast_eval_n_samples: int = None,  # fast loading some samples for testing your code
        get_name: bool = False,  # get the image file name when saveing the fused images
        reduce_label: bool = True,  # set label > 1 in mask to be 1 if is True
        crop_strategy: Literal[
            "manual", "crop", "crop_resize"
        ] = "manual",  # crop the image to the output size
        only_y_component: bool = False,  # only use the Y component of the image
        load_txt_in_ram: bool = False,  # whether to load the txt feature in ram
        only_resize: "Sequence | None" = None,  # only resize the image to the output size
    ):
        assert num_shards > 0, "num_shards should be greater than 0"
        assert (
            shard_id >= 0 and shard_id < num_shards
        ), "shard_id should be in [0, num_shards)"

        # assert reduce_label, must be True. Different datasets have different kinds of labels'
        if not reduce_label:
            logger.warning(
                "========================================================================================\n"
                + "reduce_label is set to be [red]False[/red]. Different datasets have [red]different kinds of labels[/red]\n"
                + "be sure to check the the behavior is you want before training. \n"
                + "========================================================================================\n"
            )
            input(
                "i am sure using labels with different values but assigned to different classes. Press any key to continue..."
            )

            ## TODO: assign to be a unified idx2label

        # to fit deepspeed in `accelerator.prepare`
        self.batch_size = batch_size

        logger.info(
            f"{__class__.__name__}: crop strategy is [green]{crop_strategy}[/green]"
        )

        for k, v in base_dirs.items():
            base_dirs[k.lower()] = base_dirs[k]

        each_dataset_kwargs = dict(
            mode=mode,
            output_size=output_size,
            with_mask=with_mask,
            batch_size=batch_size,
            shard_id=shard_id,
            num_shards=num_shards,
            n_thread=n_thread,
            device=device,
            shuffle=shuffle_in_dataset,
            fast_eval_n_samples=fast_eval_n_samples,
            get_name=get_name,
            reduce_label=reduce_label,
            crop_strategy=crop_strategy,
            only_y_component=only_y_component,
            with_txt_feature=with_txt_feature,
            only_resize=only_resize,
        )

        self.base_dirs = base_dirs
        self.n_dataset = 0
        self.shuffle_datasets = (
            random_datasets_getitem  # TODO: support shuffle the task_datasets.
        )
        self.num_shards = num_shards
        self.shard_id = shard_id
        # if random_datasets_getitem:
        #     logger.warning(
        #         "`random_datasets_getitem` is not supported yet. The datasets will be loaded sequentially."
        #     )
        self.get_name = get_name
        self.with_mask = with_mask
        self.batch_size = batch_size
        self._pipes = []
        self._loaders = []
        self.datasets_len = 0

        if "llvip" in base_dirs:
            LLVIP_loader = LLVIPDALIPipeLoader(
                base_dir=base_dirs["llvip"], **each_dataset_kwargs
            )
            self.LLVIP_pipe = LLVIP_loader.pipe
            pipe_len = LLVIP_loader.n_samples
            self._pipes.append(self.LLVIP_pipe)
            self._loaders.append(LLVIP_loader)

            self.datasets_len += pipe_len
            self.n_dataset += 1
            logger.info("LLVIP dataset loaded.")
        else:
            self.LLVIP_pipe = None

        if "msrs" in base_dirs:
            MSRS_loader = MSRSDALIPipeLoader(
                base_dir=base_dirs["msrs"],
                load_txt_feat_in_ram=load_txt_in_ram,
                **each_dataset_kwargs,
            )
            self.MSRS_pipe = MSRS_loader.pipe
            pipe_len = MSRS_loader.n_samples
            self._pipes.append(self.MSRS_pipe)
            self._loaders.append(MSRS_loader)

            self.datasets_len += pipe_len
            self.n_dataset += 1
            logger.info("MSRS dataset loaded.")
        else:
            self.MSRS_pipe = None

        if "m3fd" in base_dirs:
            M3FD_loader = M3FDDALIPipeLoader(
                base_dir=base_dirs["m3fd"], **each_dataset_kwargs
            )
            self.M3FD_pipe = M3FD_loader.pipe
            pipe_len = M3FD_loader.n_samples
            self._pipes.append(self.M3FD_pipe)
            self._loaders.append(M3FD_loader)

            self.datasets_len += pipe_len
            self.n_dataset += 1
            logger.info("M3FD dataset loaded.")
        else:
            self.M3FD_pipe = None

        if "roadscene_tno_joint" in base_dirs:
            ds = TNODataset(
                base_dir=base_dirs["roadscene_tno_joint"],
                mode=mode,
                device=device,
                size=output_size,
                no_split=False if mode == "train" else True,
                duplicate_vis_channel=True,
                fast_eval_n_samples=fast_eval_n_samples,
                get_name=get_name,
                with_txt_feature=with_txt_feature,
                only_resize=only_resize,
                get_downsampled_vis=False,
                output_none_mask=True,
            )
            # if num_shards > 1:
            #     sampler = DistributedSampler(ds,
            #                                  shuffle=shuffle_in_dataset,
            #                                  rank=shard_id)
            # else:
            #     sampler = None
            self.RT_pipe = DataLoader(
                ds,
                batch_size,
                num_workers=0,  # avoid the n_works > 0 raise cuda_stream re-init error
                pin_memory=False,
                shuffle=shuffle_in_dataset,  # if num_shards == 1 else None,
                # sampler=sampler,
            )
            self._loaders.append(self.RT_pipe)
            self.n_dataset += 1
            self.datasets_len += len(ds)
            logger.info("RoadScene and TNO datasets loaded.")
        else:
            self.RT_pipe = None

        assert self.n_dataset > 0, "at least one datasets should be provided"
        self.dataset_idx = 0

        self.loaders = self._iter_all_loaders()
        self._outting_idx = 0

    def _iter_all_loaders(self):
        itered_loaders = []
        if self.shuffle_datasets:
            shuffle(self._loaders)
            logger.debug(f"shuffle datasets: {[loader.dataset.__class__.__name__ if isinstance(loader, DataLoader) \
                                                else loader.__class__.__qualname__ for loader in self._loaders]}")
        for loader in self._loaders:
            itered_loaders.append(iter(loader))

        return itered_loaders

    def __iter__(self):
        return self

    def __next__(self):
        ## sequential loading
        # logger.info(f'using dataset [{self.dataset_idx}]')
        idx = self._outting_idx
        loader = self.loaders[idx]
        try:
            return next(loader)
        except StopIteration:
            if self._outting_idx == self.n_dataset - 1:
                # reload the loaders
                self.loaders = self._iter_all_loaders()

                self._outting_idx = 0
                raise StopIteration
            self._outting_idx += 1
            return self.__next__()

    def __len__(self):
        n = self.datasets_len // (self.batch_size * self.num_shards)
        if self.datasets_len % self.batch_size != 0:
            n += 1

        return n


if __name__ == "__main__":
    from rich.progress import track

    vis_ir_loader = VISIRJointGenericLoader(
        base_dirs={
            # "llvip": "/Data3/cao/ZiHanCao/datasets/VIF-LLVIP/data",
            "msrs": "/Data3/cao/ZiHanCao/datasets/VIF-MSRS",
            # "m3fd": "/Data3/cao/ZiHanCao/datasets/VIF-M3FD",
            "roadscene_tno_joint": "/Data3/cao/ZiHanCao/datasets/VIF-RoadScene_and_TNO",
        },
        mode="test",
        with_mask=True,
        with_txt_feature=True,
        random_datasets_getitem=True,
        batch_size=1,
        shard_id=0,
        num_shards=1,
        output_size=(224, 280),
        n_thread=4,
        # only_resize=(224, 280),
        fast_eval_n_samples=80,
        get_name=True,
        reduce_label=True,
        crop_strategy="crop_resize",
        only_y_component=False,
        device="cuda:0",
    )
    torch.cuda.set_device(0)

    from tqdm import tqdm

    iter_idx = 0
    while True:
        print(f"loop iter {iter_idx}")
        for i, data in tqdm(enumerate(vis_ir_loader), total=len(vis_ir_loader)):
            vi, ir, mask, gt, name = (
                data["vi"],
                data["ir"],
                data["mask"],
                data["gt"],
                data["name"],
            )
            txt = data["txt"]
            assert (
                vi.shape[-2:] == ir.shape[-2:]
            ), f"vi and ir should have the same shape, but got {vi.shape} and {ir.shape}"
            assert vi.ndim == 4
            assert ir.ndim == 4
            # assert mask.ndim == 4
            assert gt.ndim == 4
            assert (
                txt.ndim == 3 and txt.shape[-2:] == torch.Size([512, 1024])
            ), f"txt should be a 3D tensor with the shape of (512, 1024), but got {txt.shape}"
            # print(vi.device, ir.device, mask.device, gt.device, txt.device)

        iter_idx += 1
