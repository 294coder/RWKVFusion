import torch
import torch.utils.data as data
import torchvision.transforms as T
import cv2
import numpy as np
import h5py
from safetensors import safe_open
from safetensors.torch import load_file
from typing import List, Tuple, Optional, Union

from utils import easy_logger

logger = easy_logger(func_name="WV3 dataset")


class Identity:
    def __call__(self, *args):
        # args is a tuple
        # return is also a tuple
        return args


# WV3_GT_MEAN = [0.13435693, 0.15736127, 0.19913845, 0.17137502, 0.13985378,
#                0.16384054, 0.21204206, 0.1553395]
# WV3_GT_STD = [0.04436018, 0.07571019, 0.12324945, 0.12895705, 0.12202228,
#               0.10989053, 0.13726164, 0.1000899]
# WV3_PAN_MEAN = [0.19546394]
# WV3_PAN_STD = [0.11308921]
#
# QB_GT_MEAN = [0.08384636, 0.10903837, 0.06165434, 0.07774738]
# QB_GT_STD = [0.04095699, 0.07015568, 0.05757316, 0.07613233]
# QB_PAN_MEAN = [0.0815676]
# QB_PAN_STD = [0.05739738]


class WV3Datasets(data.Dataset):
    def __init__(
        self,
        file: Union[h5py.File, str, dict],
        aug_prob=0.0,
        hp_ksize=(5, 5),
        hp=False,
        norm_range=True,
        full_res=False,
        txt_file=None,
        txt_feature_online_load: bool = False,
    ):
        """

        :param d: h5py.File or dict
        :param aug_prob: augmentation probability
        :param hp: high pass for ms and pan. x = x - cv2.boxFilter(x)
        :param hp_ksize: cv2.boxFiler kernel size
        :param norm_range: normalize data range
        """
        super(WV3Datasets, self).__init__()
        # FIXME: should pass @path raher than @file which is h5py.File object to avoid can not be pickled error

        self.full_res = full_res
        self.with_txt = True if txt_file is not None else False
        self.txt_feature_online_load = txt_feature_online_load

        if isinstance(file, (str, h5py.File)):
            if isinstance(file, str):
                file = h5py.File(file)
            print(
                "warning: when @file is a h5py.File object, it can not be pickled.",
                "try to set DataLoader number_worker to 0",
            )
        if not full_res:
            self.gt, self.ms, self.lms, self.pan = self.get_divided(file)
            logger.print("datasets shape:")
            logger.print("{:^20}{:^20}{:^20}{:^20}".format("pan", "ms", "lms", "gt"))
            logger.print(
                "{:^20}{:^20}{:^20}{:^20}".format(
                    str(tuple(self.pan.shape)),
                    str(tuple(self.ms.shape)),
                    str(tuple(self.lms.shape)),
                    str(tuple(self.gt.shape)),
                )
            )
        else:
            self.ms, self.lms, self.pan = self.get_divided(file, True)
            logger.print("datasets shape:")
            logger.print("{:^20}{:^20}{:^20}".format("pan", "ms", "lms"))
            logger.print(
                "{:^20}{:^20}{:^20}".format(
                    str(tuple(self.pan.shape)),
                    str(tuple(self.ms.shape)),
                    str(tuple(self.lms.shape)),
                )
            )

        self.size = self.ms.shape[0]

        # load txt features
        if self.with_txt:
            logger.info(
                f"loading txt features from {txt_file} with mode {'online' if self.txt_feature_online_load else 'offline'} ..."
            )
            if self.txt_feature_online_load:
                self.txt_features_f = self.get_txt_features(txt_file)
                assert len(self.txt_features_f.keys()) == self.size, (
                    f"txt features length should be equal to dataset length",
                    f"but got {len(self.txt_features_f.keys())} and {self.size}",
                )
            else:
                self.txt_features = self.get_txt_features(txt_file)
                assert len(self.txt_features) == self.size, (
                    f"txt features length should be equal to dataset length",
                    f"but got {len(self.txt_features)} and {self.size}",
                )
                logger.info(
                    f"txt features loaded. every sample has a txt feature shaped as",
                    f"{self.txt_features[next(iter(self.txt_features.keys()))].shape}",
                )

        # highpass filter
        self.hp = hp
        self.hp_ksize = hp_ksize
        if hp and hp_ksize is not None:
            self.group_high_pass(hp_ksize)

        # to tensor
        if norm_range:

            def norm_func(x):
                x = x / 2047.0
                return x
        else:

            def norm_func(x):
                return x

        self.pan = norm_func(self.pan)
        self.ms = norm_func(self.ms)
        self.lms = norm_func(self.lms)
        if not full_res:
            self.gt = norm_func(self.gt)

        # geometrical transformation
        self.aug_prob = aug_prob
        self.geo_trans = (
            T.Compose(
                [T.RandomVerticalFlip(p=aug_prob), T.RandomHorizontalFlip(p=aug_prob)]
            )
            if aug_prob != 0.0
            else Identity()
        )

    def get_txt_features(self, txt_file) -> dict[str, np.ndarray]:
        if self.txt_feature_online_load:
            return safe_open(txt_file, framework="torch")
        else:
            return load_file(txt_file)

    @staticmethod
    def get_divided(d, full_resolution=False):
        if not full_resolution:
            return (
                torch.tensor(d["gt"][:], dtype=torch.float32),  # .clip(0, 2047),
                torch.tensor(d["ms"][:], dtype=torch.float32),  # .clip(0, 2047),
                torch.tensor(d["lms"][:], dtype=torch.float32),  # .clip(0, 2047),
                torch.tensor(d["pan"][:], dtype=torch.float32),  # .clip(0, 2047),
            )
        else:
            return (
                torch.tensor(d["ms"][:], dtype=torch.float32),  # .clip(0, 2047),
                torch.tensor(d["lms"][:], dtype=torch.float32),  # .clip(0, 2047),
                torch.tensor(d["pan"][:], dtype=torch.float32),  # .clip(0, 2047),
            )

    @staticmethod
    def _get_high_pass(data, k_size):
        for i, img in enumerate(data):
            hp = cv2.boxFilter(img.transpose(1, 2, 0), -1, k_size)
            if hp.ndim == 2:
                hp = hp[..., np.newaxis]
            data[i] = img - hp.transpose(2, 0, 1)
        return data

    def group_high_pass(self, k_size):
        self.ms = self._get_high_pass(self.ms, k_size)
        self.pan = self._get_high_pass(self.pan, k_size)

    def aug_trans(self, data):
        # we set a seed from the training start
        _random_state = torch.random.get_rng_state()
        data_dict = {}
        for k, d in data.items():
            torch.random.set_rng_state(_random_state)
            d = self.geo_trans(d)
            data_dict[k] = d
        return data_dict

    def __getitem__(self, item):
        outp = {"ms": self.ms[item], "lms": self.lms[item], "pan": self.pan[item]}
        if hasattr(self, "gt"):
            outp["gt"] = self.gt[item]
        if self.aug_prob != 0.0:
            outp = self.aug_trans(outp)
        if self.with_txt:
            if self.txt_feature_online_load:
                outp["txt"] = self.txt_features_f.get_tensor(str(item))[0].type(
                    torch.float32
                )
            else:
                outp["txt"] = self.txt_features[str(item)][0].type(torch.float32)

        return outp

    def __len__(self):
        return self.size

    def __repr__(self):
        return (
            f"num: {self.size} \n "
            f"augmentation: {self.geo_trans} \n"
            f"get high pass ms and pan: {self.hp} \n "
            f"filter kernel size: {self.hp_ksize}"
        )


def make_datasets(
    path, split_ratio=0.8, hp=True, seed=2022, aug_probs: Tuple = (0.0, 0.0)
):
    """
    if your dataset didn't split before, use this function will split your dataset into two part,
    which are train and validate datasets.
    :param device: device
    :param path: datasets path
    :param split_ratio: train validate split ratio
    :param hp: get high pass data, only works for ms and pan data
    :param seed: split data random state
    :param aug_probs: augmentation probabilities, type List
    :return: List[datasets]
    """
    d = h5py.File(path)
    ds = [
        torch.tensor(d["gt"]),
        torch.tensor(d["ms"]),
        torch.tensor(d["lms"]),
        torch.tensor(d["pan"]),
    ]
    n = ds[0].shape[0]
    s = int(n * split_ratio)
    random_perm = np.arange(n)
    np.random.seed(seed)
    np.random.shuffle(random_perm)

    train_set = {}
    val_set = {}
    for i, name in enumerate(["gt", "ms", "lms", "pan"]):
        ds[i] = ds[i][random_perm]
        train_set[name] = ds[i][:s]
        val_set[name] = ds[i][s:]
    train_ds = WV3Datasets(train_set, hp=hp, aug_prob=aug_probs[0])
    val_ds = WV3Datasets(val_set, hp=hp, aug_prob=aug_probs[1])
    return train_ds, val_ds


if __name__ == "__main__":
    from tqdm import tqdm

    path = "/Data3/cao/ZiHanCao/datasets/pansharpening/qb/training_qb/train_qb.h5"
    txt_path = "/Data3/cao/ZiHanCao/datasets/pansharpening/qb/training_qb/t5_feature_qb_train.safetensors"

    train_ds = WV3Datasets(
        path, full_res=False, txt_file=txt_path, txt_feature_online_load=True
    )
    train_dl = data.DataLoader(
        train_ds,
        32,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
    )
    for data in tqdm(train_dl):
        lms = data["lms"]
        pan = data["pan"]
        txt = data["txt"]

        assert lms.shape[-2:] == (64, 64), f"{lms.shape}"
        assert pan.shape[-2:] == (64, 64), f"{pan.shape}"
        assert txt.shape[-2:] == (512, 512), f"{txt.shape}"

        # import matplotlib.pyplot as plt
        # import seaborn as sns

        # plot_gt = gt[2, :3].permute(1, 2, 0) * torch.tensor(
        #     train_ds.gt_std[:3]
        # ) + torch.tensor(train_ds.gt_mean[:3])
        # ori_gt = gt * torch.tensor(train_ds.gt_std).view(1, 4, 1, 1) + torch.tensor(
        #     train_ds.gt_mean
        # ).view(1, 4, 1, 1)
        # print(pan.mean(), ms.mean(), lms.mean(), ori_gt.mean())
        # print(pan.min(), ms.min(), lms.min(), ori_gt.min())
        # print(pan.max(), ms.max(), lms.max(), ori_gt.max())
        # plt.imshow(plot_gt)
        # plt.show()
        # sns.distplot(gt.flatten().numpy())
        # plt.show()
        # sns.distplot(ori_gt.flatten().numpy())
        # plt.show()
        # # assert pan.shape[-1] == 64 and ms.shape[-1] == 16 and lms.shape[-1] == 64 and gt.shape[
        # # -1] == 64, f'{pan.shape, ms.shape, lms.shape, gt.shape}'
        # break
