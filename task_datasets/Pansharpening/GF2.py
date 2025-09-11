import torch
import torch.utils.data as data
import torchvision.transforms as T
import cv2
import numpy as np
import h5py
from safetensors import safe_open
from safetensors.numpy import load_file
from typing import List, Tuple, Optional

from utils import easy_logger

logger = easy_logger(func_name="GF2")


class Identity:
    def __call__(self, *args):
        # args is a tuple
        # return is also a tuple
        return args


class GF2Datasets(data.Dataset):
    def __init__(
        self,
        d,
        aug_prob=0.0,
        hp=False,
        hp_ksize=(5, 5),
        norm_range=True,
        full_res=False,
        const=1023.0,
        txt_file=None,
        txt_feature_online_load: bool = True,
    ):
        """

        :param d: h5py.File or dict or path
        :param aug_prob: augmentation probability
        :param hp: high pass for ms and pan. x = x - cv2.boxFilter(x)
        :param hp_ksize: cv2.boxFiler kernel size
        :param norm_range: normalize data range
        """
        super(GF2Datasets, self).__init__()
        # FIXME: should pass @path rather than @file which is h5py.File object to avoid can not be pickled error

        self.full_res = full_res
        self.with_txt = True if txt_file is not None else False
        self.txt_feature_online_load = txt_feature_online_load

        if isinstance(d, (str, h5py.File)):
            if isinstance(d, str):
                d = h5py.File(d)
            logger.info(
                "warning: when @file is a h5py.File object, it can not be pickled."
                + "try to set DataLoader number_worker to 0",
            )
        if not full_res:
            self.gt, self.ms, self.lms, self.pan = self.get_divided(d)
            logger.info("datasets shape:")
            logger.info("{:^20}{:^20}{:^20}{:^20}".format("pan", "ms", "lms", "gt"))
            logger.info(
                "{:^20}{:^20}{:^20}{:^20}".format(
                    str(self.pan.shape),
                    str(self.ms.shape),
                    str(self.lms.shape),
                    str(self.gt.shape),
                )
            )
        else:
            self.ms, self.lms, self.pan = self.get_divided(d, True)
            logger.info("datasets shape:")
            logger.info("{:^20}{:^20}{:^20}".format("pan", "ms", "lms"))
            logger.info(
                "{:^20}{:^20}{:^20}".format(
                    str(self.pan.shape), str(self.ms.shape), str(self.lms.shape)
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
                    f"but got {len(self.txt_features)} and {self.size}",
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
                # return torch.tensor(x) / 2047.
                return torch.tensor(x, dtype=torch.float32) / const

        else:

            def norm_func(x):
                return torch.tensor(x, dtype=torch.float32)

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

    @staticmethod
    def get_divided(d, full_resolution=False):
        if not full_resolution:
            return (
                np.asarray(d["gt"]),
                np.asarray(d["ms"]),
                np.asarray(d["lms"]),
                np.asarray(d["pan"]),
            )
        else:
            return (np.asarray(d["ms"]), np.asarray(d["lms"]), np.asarray(d["pan"]))

    def get_txt_features(self, txt_file) -> dict[str, np.ndarray]:
        if self.txt_feature_online_load:
            return safe_open(txt_file, framework="numpy")
        else:
            return load_file(txt_file)

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

    def aug_trans(self, *data):
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
            outp = self.aug_trans(*outp)
        if self.with_txt:
            if self.txt_feature_online_load:
                outp["txt"] = torch.from_numpy(
                    self.txt_features_f.get_tensor(str(item))[0].astype(np.float32)
                )
            else:
                outp["txt"] = torch.from_numpy(
                    self.txt_features[str(item)][0].astype(np.float32)
                )

        # output includes: ms, lms, pan, gt, txt
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


if __name__ == "__main__":
    import torch.utils.data as D

    path = "/home/ZiHanCao/datasets/pansharpening/gf/training_gf2/train_gf2.h5"
    d = h5py.File(path)
    ds = GF2Datasets(d, norm_range=True, hp=False)
    dl = D.DataLoader(ds, batch_size=16, num_workers=6)
    for gt, ms, lms, pan in dl:
        print(gt.shape, ms.shape, lms.shape, pan.shape, sep="\n")
        break
