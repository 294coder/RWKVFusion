import argparse
import importlib
import json
import os
import os.path as osp
import random
import zipfile
from collections.abc import Mapping
from contextlib import contextmanager
from functools import lru_cache
from io import BytesIO
from typing import Any, Iterable, Mapping, Union

import h5py
import numpy as np
import shortuuid
import torch
import torch.distributed as dist
import yaml
from omegaconf import DictConfig, ListConfig, OmegaConf
from PIL import Image
from torch.backends import cudnn

# * ========================================================================
# * some utilities
# * ========================================================================


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


def is_none(val):
    return val in ("none", "None", "NONE", None)


def set_all_seed(seed=2022):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    cudnn.deterministic = True  # not all operations are deterministic in Pytorch
    cudnn.benchmark = False


def to_numpy(*args):
    l = []
    for i in args:
        if isinstance(i, torch.Tensor):
            l.append(i.detach().cpu().numpy())
    return l


def to_tensor(*args, device, dtype):
    out = []
    for a in args:
        out.append(torch.as_tensor(a, dtype=dtype, device=device))
    return out


def args_no_str_none(value: str) -> "str | None":
    if value.lower() == "none":
        return None
    return value


def to_device(*args, device):
    out = []
    for a in args:
        out.append(a.to(device) if torch.is_tensor(a) else a)
    return out


def h5py_to_dict(file: h5py.File, keys=None) -> dict[str, np.ndarray]:
    """Get all content in a h5py file into a dict contains key and values

    Args:
        file (h5py.File): h5py file
        keys (list, optional): h5py file keys used to extract values.
        Defaults to ["ms", "lms", "pan", "gt"].

    Returns:
        dict[str, np.ndarray]:
    """
    d = {}
    if keys is None:
        keys = list(file.keys())
    for k in keys:
        print(f"reading key {k} array from h5 file...")
        d[k] = file[k][:]
    return d


def dict_to_str(d, decimals=4):
    n = len(d)

    # func = lambda k, v: f"{k}: {torch.round(v, decimals=decimals).item() if isinstance(v, torch.Tensor) else round(v, decimals)}"
    def func(k, v):
        if isinstance(v, torch.Tensor):
            return f"{k}: {round(v.item(), decimals)}"
        elif isinstance(v, np.ndarray):
            return f"{k}: {np.round(v, decimals=decimals)}"
        elif isinstance(v, (float, int)):
            return f"{k}: {round(v, decimals)}"
        else:
            raise ValueError(f"Unsupported type: {type(v)}")

    s = ""
    for i, (k, v) in enumerate(d.items()):
        s += func(k, float(v)) + (", " if i < n - 1 else "")
    return s


def prefixed_dict_key(d, prefix, sep="_"):
    # e.g.
    # SSIM -> train_SSIM
    d2 = {}
    for k, v in d.items():
        d2[prefix + sep + k] = v
    return d2


# deprecated
class CheckPointManager(object):
    def __init__(
        self,
        model: torch.nn.Module,
        save_path: str,
        save_every_eval: bool = False,
        verbose: bool = True,
    ):
        """
        manage model checkpoints
        Args:
            model: nn.Module, can be single node model or multi-nodes model
            save_path: str like '/home/model_ckpt/resnet.pth' or '/home/model_ckpt/exp1' when @save_every_eval
                       is False or True
            save_every_eval: when False, save params only when ep_loss is less than optim_loss.
                            when True, save params every eval epoch
            verbose: print out all information

        e.g.
        @save_every_eval=False, @save_path='/home/ckpt/resnet.pth'
        weights will be saved like
        -------------
        /home/ckpt
        |-resnet.pth
        -------------

        @save_every_eval=True, @save_path='/home/ckpt/resnet'
        weights will be saved like
        -------------
        /home/ckpt
        |-resnet
            |-ep_20.pth
            |-ep_40.pth
        -------------

        """
        self.model = model
        self.save_path = save_path
        self.save_every_eval = save_every_eval
        self._optim_loss = torch.inf
        self.verbose = verbose

        self.check_path_legal()

    def check_path_legal(self):
        if self.save_every_eval:
            if not os.path.exists(self.save_path):
                os.makedirs(self.save_path)
        else:
            assert self.save_path.endswith(".pth")
            par_dir = os.path.dirname(self.save_path)
            if not os.path.exists(par_dir):
                os.makedirs(par_dir)

    def save(
        self,
        ep_loss: Union[float, torch.Tensor] = None,
        ep: int = None,
        extra_saved_dict: dict = None,
    ):
        """

        Args:
            ep_loss: should be set when @save_every_eval=False
            ep: should be set when @save_every_eval=True
            extra_saved_dict: a dict which contains other information you want to save with model
                            e.g. {'optimizer_ckpt': op_ckpt, 'time': '2023/1/21'}

        Returns:

        """
        if isinstance(ep_loss, torch.Tensor):
            ep_loss = ep_loss.item()

        saved_dict = {}
        if not self.save_every_eval:
            assert ep_loss is not None
            if ep_loss < self._optim_loss:
                self._optim_loss = ep_loss
                path = self.save_path
                saved_dict["optim_loss"] = ep_loss
            else:
                print(
                    "optim loss: {}, now loss: {}, not saved".format(
                        self._optim_loss, ep_loss
                    )
                )
                return
        else:
            assert ep is not None
            path = os.path.join(self.save_path, "ep_{}.pth".format(ep))

        if extra_saved_dict is not None:
            assert "model" not in list(saved_dict.keys())
            saved_dict = extra_saved_dict

        try:
            saved_dict["model"] = self.model.module.state_dict()
        except:
            saved_dict["model"] = self.model.state_dict()

        torch.save(saved_dict, path)

        if self.verbose:
            print(
                "saved params contains\n",
                *[
                    "\t -{}: {}\n".format(k, v if k != "model" else "model params")
                    for k, v in saved_dict.items()
                ],
                "saved path: {}".format(path),
            )


# * ========================================================================
# * distribution utils using torch.distributed
# * ========================================================================


@lru_cache(maxsize=1)
def get_process_index():
    if torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    else:
        return 0


def is_main_process(func=None):
    """
    check if current process is main process in ddp
    warning: if not in ddp mode, always return True
    :return:
    """

    def _is_main_proc():
        return get_process_index() == 0

    if func is None:
        return _is_main_proc()
    else:

        def warp_func(*args, **kwargs):
            if _is_main_proc():
                return func(*args, **kwargs)
            else:
                return None

        return warp_func


def print_args(args):
    d = args.__dict__
    for k, v in d.items():
        print(f"{k}: {v}")


def yaml_tuple_constructor(loader, node):
    return tuple(loader.construct_sequence(node))


yaml.add_constructor("tag:yaml.org,2002:tuple", yaml_tuple_constructor)


def yaml_load(name, base_path="./configs", end_with="_config.yaml"):
    if base_path is not None:
        path = osp.join(base_path, (name + end_with) if end_with is not None else name)
    else:
        path = name
    if osp.exists(path):
        f = open(path)
        cont = f.read()
        return yaml.load(cont, Loader=yaml.FullLoader)
    else:
        print(f"configuration file {path} not exists")
        raise FileNotFoundError(f"file not exists: {path}")


def json_load(name, base_path="./configs"):
    path = osp.join(base_path, name + "_config.json")
    with open(path) as f:
        return json.load(f)


def config_py_load(name, base_path="configs"):
    args = importlib.import_module(f".{name}_config", package=base_path)
    return args.config


# * ========================================================================
# * namespace utilities
# * ========================================================================


class NameSpace(Mapping):
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    @property
    def attrs(self):
        return self.__dict__

    def to_dict(self):
        d = self.attrs
        if isinstance(d, dict):
            out = {}
            for k, v in d.items():
                if isinstance(v, NameSpace):
                    out[k] = v.to_dict()
                elif isinstance(v, list):
                    lst = []
                    for i in v:
                        if isinstance(i, NameSpace):
                            lst.append(i.to_dict())
                        else:
                            lst.append(i)
                    out[k] = lst
                else:
                    out[k] = v
        elif isinstance(d, list):
            out = []
            for i in d:
                if isinstance(i, NameSpace):
                    out.append(i.to_dict())
                else:
                    out.append(i)
        else:
            out = d
        return out

    def __repr__(self, d=None, nprefix=0):
        repr_str = ""
        if d is None:
            d = self.attrs
        for k, v in d.items():
            if isinstance(v, NameSpace):
                repr_str += (
                    "  " * nprefix
                    + f"{k}: \n"
                    + f"{self.__repr__(v.attrs, nprefix + 1)}"
                )
            else:
                repr_str += "  " * nprefix + f"{k}: {v}\n"

        return repr_str

    def __getitem__(self, item):
        if item not in self.attrs:
            raise KeyError(f"{item} not in {self.attrs}")
        return self.attrs[item]

    def __setitem__(self, key, value):
        self.attrs[key] = value

    def __contains__(self, key: object) -> bool:
        keys_split = key.split(".")
        current = self
        for k in keys_split:
            if isinstance(current, NameSpace) and k in current.attrs:
                current = current.attrs[k]
            elif isinstance(current, dict) and k in current:
                current = current[k]
            else:
                return False
        return True

    def __iter__(self):
        """
        support dict-like iteration and unpacking operator
        for example:

            1. dict-like iteration
        >>> args = NameSpace(
        ...     a=1, b=2
        ... )
        >>> for k, v in args:
        >>>     print(k, v)
        >>> a 1
        >>> b 2

            2. unpacking operator
        >>> args = NameSpace(
        ...     a=1, b=2
        ... )
        >>> {**args}
        {'a': 1, 'b': 2}
        """
        return iter(self.to_dict())

    def __next__(self):
        return next(iter(self))

    def __len__(self):
        return len(self.__dict__)

    @classmethod
    def dict_to_namespace(cls, d: "dict | Any | list"):
        return dict_to_namespace(d)

    @classmethod
    def merge_args(cls, *args: "NameSpace"):
        return merge_args(*args)

    @classmethod
    def merge_parser_args(
        cls, parser_args: argparse.Namespace, namespace_args: "NameSpace"
    ):
        return merge_args_namespace(parser_args, namespace_args)

    @classmethod
    def merge_dict_configs(cls, dict_config: DictConfig, namespace_args: "NameSpace"):
        dict_rm_metas = OmegaConf.to_container(dict_config, resolve=True)
        args_dc = cls.dict_to_namespace(dict_rm_metas)
        return merge_args(args_dc, namespace_args)

    def default_none_getattr(self, key: str) -> "Any | None":
        return default_none_getattr(self, key)

    def default_getattr(self, key: str, default: Any) -> Any:
        return default_getattr(self, key, default)

    @classmethod
    def init_from_yaml(
        cls, path: str, base_path: str | None = None, end_with: str | None = None
    ):
        d = yaml_load(path, base_path=base_path, end_with=end_with)
        return cls.dict_to_namespace(d)

    @classmethod
    def init_from_json(cls, path: str):
        d = json_load(path)
        return cls.dict_to_namespace(d)


def default_none_getattr(args: NameSpace, key: str) -> "Any | None":
    """
    Retrieve an attribute from a NameSpace object, returning None if the attribute does not exist.

    Args:
        args (NameSpace): The NameSpace object to retrieve the attribute from.
        key (str): The key of the attribute to retrieve.

    Returns:
        Any | None: The value of the attribute if it exists, otherwise None.
    """
    _args = args
    keys = key.split(".")
    for k in keys:
        _args = getattr(_args, k, None)
        if _args is None:
            return None

    return _args


def default_getattr(args: NameSpace, key: str, default: Any) -> Any:
    """
    Retrieve an attribute from a NameSpace object, returning a default value if the attribute does not exist.

    Args:
        args (NameSpace): The NameSpace object to retrieve the attribute from.
        key (str): The key of the attribute to retrieve.
        default (Any): The default value to return if the attribute does not exist.

    Returns:
        Any: The value of the attribute if it exists, otherwise the default value.
    """
    _args = args
    keys = key.split(".")
    for k in keys:
        _args = getattr(_args, k, default)
        if _args is default and (not isinstance(_args, NameSpace)):
            return default

    return _args


def dict_to_namespace(d: "dict | Any | list"):
    """
    convert a yaml-like configuration (dict) to namespace-like class

    e.g.
    {'lr': 1e-3, 'path': './datasets/train_wv3.h5'} ->
    NameSpace().lr = 1e-3, NameSpace().path = './datasets/train_wv3.h5'

    Warning: the value in yaml-like configuration should not be another dict
    :param d:
    :return:
    """
    namespace = NameSpace()
    if isinstance(d, dict):
        for k, v in d.items():
            if isinstance(v, dict):
                setattr(namespace, k, dict_to_namespace(v))
            elif isinstance(v, list):
                lst = []
                for i in v:
                    lst.append(dict_to_namespace(i))
                setattr(namespace, k, lst)
            else:
                setattr(namespace, k, v)
        return namespace
    elif isinstance(d, list):
        lst = []
        for i in d:
            lst.append(dict_to_namespace(i))
        return lst
    else:
        return d


def merge_args(*args: "NameSpace"):
    """
    merge multiple NameSpace objects into one
    key/value in latter args will cover former args' key/value
    """
    namespace = NameSpace()
    dicts = [d.to_dict() for d in args]
    d = {}
    for di in dicts:
        d.update(di)

    namespace = dict_to_namespace(d)
    return namespace


def merge_args_namespace(
    parser_args: argparse.Namespace | DictConfig, namespace_args: NameSpace
):
    """
    merge parser_args and self-made class _NameSpace configurations together for better
    usage.
    return args that support dot its member, like args.optimizer.lr
    :param parser_args:
    :param namespace_args:
    :return:
    """
    # namespace_args.__dict__.update(parser_args.__dict__)
    namespace_d = namespace_args.__dict__
    for k, v in parser_args.__dict__.items():
        if not (k in namespace_d.keys() and v is None):
            setattr(namespace_args, k, v)

    return namespace_args


def flatten_namespace_first_level(args: NameSpace):
    """flatten the first level of namespace"""
    d = {}
    for k, v in args.to_dict().items():
        if isinstance(v, dict):
            for k1, v1 in v.items():
                d[k1] = v1
        else:
            d[k] = v

    return NameSpace.dict_to_namespace(d)


def yaml_load_tuple_in_hydra():
    def resolve_tuple(*args):
        return tuple(args)

    OmegaConf.register_new_resolver("tuple", resolve_tuple)


# FIXME: this function is not working
def omega_config_listconfig_to_list(
    cfg: DictConfig | ListConfig,
) -> DictConfig | ListConfig:
    """
    Convert ListConfig in OmegaConf to python list recursively
    Keep DictConfig unchanged but convert nested ListConfig to list
    """
    if isinstance(cfg, ListConfig):
        # Convert ListConfig to list
        lst = []
        for item in cfg:
            if isinstance(item, ListConfig):
                # Recursively convert nested ListConfig
                lst.append(omega_config_listconfig_to_list(item))
            elif isinstance(item, DictConfig):
                # Keep DictConfig but process its contents
                for k in item:
                    try:
                        v = item[k]
                        if isinstance(v, ListConfig):
                            cfg[item][k] = omega_config_listconfig_to_list(v)
                        elif isinstance(v, DictConfig):
                            cfg[item][k] = omega_config_listconfig_to_list(v)
                        else:
                            cfg[item][k] = v
                    except Exception as e:
                        # Skip unsupported interpolation types
                        print(f"Warning: Skipping key {k} due to error: {str(e)}")
                        continue
                lst.append(cfg)
            else:
                lst.append(item)
        return lst
    elif isinstance(cfg, DictConfig):
        # Process DictConfig without converting it
        for k in cfg:
            try:
                v = cfg[k]
                if isinstance(v, ListConfig):
                    #!! this setitem is not working
                    #!! it will be converted to ListConfig again
                    cfg[k] = omega_config_listconfig_to_list(v)
                elif isinstance(v, DictConfig):
                    cfg[k] = omega_config_listconfig_to_list(v)
            except Exception as e:
                # Skip unsupported interpolation types
                print(f"Warning: Skipping key {k} due to error: {str(e)}")
                continue
        return cfg

    return cfg


# * ========================================================================
# * other utilities
# * ========================================================================


def generate_id(length: int = 8) -> str:
    # ~3t run ids (36**8)
    run_gen = shortuuid.ShortUUID(alphabet=list("0123456789abcdefghijklmnopqrstuvwxyz"))
    return str(run_gen.random(length))


def find_weight(weight_dir="./weight/", id=None, func=None):
    """
    return weight absolute path referring to id
    Args:
        weight_dir: weight dir that saved weights
        id: weight id
        func: split string function

    Returns: str, absolute path

    """
    assert id is not None, "@id can not be None"
    weight_list = os.listdir(weight_dir)
    if func is None:
        func = lambda x: x.split(".")[0].split("_")[-1]
    for id_s in weight_list:
        only_id = func(id_s)
        if only_id == id:
            return os.path.abspath(os.path.join(weight_dir, id_s))
    print(f"can not find {id}")
    return None


def clip_dataset_into_small_patches(
    file: h5py.File,
    patch_size: int,
    up_ratio: int,
    ms_channel: int,
    pan_channel: int,
    dataset_keys: Union[list[str], tuple[str]] = ("gt", "ms", "lms", "pan"),
    save_path: str = "./data/clip_data.h5",
):
    """
    clip patches at spatial dim
    Args:
        file: h5py.File of original dataset
        patch_size: ms clipped size
        up_ratio: shape of lms divide shape of ms
        ms_channel:
        pan_channel:
        dataset_keys: similar to [gt, ms, lms, pan]
        save_path: must end with h5

    Returns:

    """
    unfold_fn = lambda x, c, ratio: (
        torch.nn.functional.unfold(
            x, kernel_size=patch_size * ratio, stride=patch_size * ratio
        )
        .transpose(1, 2)
        .reshape(-1, c, patch_size * ratio, patch_size * ratio)
    )

    assert len(dataset_keys) == 4, "length of @dataset_keys should be 4"
    assert save_path.endswith("h5"), "saved file should end with h5 but get {}".format(
        save_path.split(".")[-1]
    )
    gt = unfold_fn(torch.tensor(file[dataset_keys[0]][:]), ms_channel, up_ratio)
    ms = unfold_fn(torch.tensor(file[dataset_keys[1]][:]), ms_channel, 1)
    lms = unfold_fn(torch.tensor(file[dataset_keys[2]][:]), ms_channel, up_ratio)
    pan = unfold_fn(torch.tensor(file[dataset_keys[3]][:]), pan_channel, up_ratio)

    print("clipped datasets shape:")
    print("{:^20}{:^20}{:^20}{:^20}".format(*[k for k in dataset_keys]))
    print(
        "{:^20}{:^20}{:^20}{:^20}".format(
            str(gt.shape), str(ms.shape), str(lms.shape), str(pan.shape)
        )
    )

    base_path = os.path.dirname(save_path)
    if not os.path.exists(base_path):
        os.makedirs(base_path)
        print(f"make dir {base_path}")

    save_file = h5py.File(save_path, "w")
    for k, data in zip(dataset_keys, [gt, ms, lms, pan]):
        save_file.create_dataset(name=k, data=data)
        print(f"create data {k}")

    file.close()
    save_file.close()
    print("file closed")


# *==============================================================
# * distributed utils
# *==============================================================


def dist_gather_object(obj, n_ranks=1, dest=0, all_gather=False):
    def _iter_tensor_to_rank(rank_obj, dest=0):
        if isinstance(rank_obj, dict):
            for k, v in rank_obj.items():
                if isinstance(v, torch.Tensor):
                    rank_obj[k] = v.to(dest)
                elif isinstance(v, Iterable):
                    rank_obj[k] = _iter_tensor_to_rank(v, dest)
        elif isinstance(rank_obj, (list, tuple)):
            if isinstance(rank_obj, tuple):
                rank_obj = list(rank_obj)
            for i, v in enumerate(rank_obj):
                if isinstance(v, torch.Tensor):
                    rank_obj[i] = v.to(dest)
                elif isinstance(v, Iterable):
                    rank_obj[i] = _iter_tensor_to_rank(v, dest)
        elif isinstance(rank_obj, torch.Tensor):
            rank_obj = rank_obj.to(dest)

        return rank_obj

    if n_ranks == 1:
        return obj
    elif n_ranks > 1:
        rank_objs = [None] * n_ranks
        if all_gather:
            # all proc to proc dest
            dist.all_gather_object(rank_objs, obj)
            # if is_main_process():
            #     _scattered_objs_lst = [rank_objs] * n_ranks
            # else:
            #     _scattered_objs_lst = [None] * n_ranks
            # received_objs = [None]
            # dist.scatter_object_list(received_objs, _scattered_objs_lst)
            rank_objs = _iter_tensor_to_rank(rank_objs, dest=dest)
        else:
            dist.gather_object(obj, rank_objs if is_main_process() else None, dest)
            if is_main_process():
                rank_objs = _iter_tensor_to_rank(rank_objs, dest)
        return rank_objs
    else:
        raise ValueError("n_ranks should be greater than 0")


@contextmanager
def save_imgs_in_zip(
    zipfile_name: str, mode="w", verbose: bool = False, save_file_ext: str = "jpeg"
):
    """save images to a zip file

    Args:
        zipfile_name (str): zip filename
        mode (str, optional): mode to write in. Defaults to "w".
        verbose (bool, optional): print out. Defaults to False.
        save_file_ext (str, optional): image extension in the zip file. Defaults to "jpeg".

    Yields:
        callable: a function to save image

    Examples::

        with (
            save_imgs_in_zip(
                "zip_file.zip"
            ) as add_image
        ):
            (
                img,
                img_name,
            ) = get_img()
            add_image(
                img,
                img_name,
            )

    :ref: `add_image`

    """
    logger = easy_logger()

    # save_file_ext = save_file_ext.upper()
    zf = zipfile.ZipFile(
        zipfile_name, mode=mode, compression=zipfile.ZIP_DEFLATED, compresslevel=9
    )
    bytes_io = BytesIO()
    # jpg compression
    _jpg_quality = 100  # 95 if save_file_ext in ["jpeg", "jpg", "JPG", "JPEG"] else 100

    try:
        logger.info(f"zip file will be saved at {zipfile_name}")

        def to_bytes(image_data, image_name):
            batched_image_bytes = []

            if image_data.ndim == 4:  # batched rgb images
                assert isinstance(image_name, list), "image_name should be a list"
                assert image_data.shape[0] == len(
                    image_name
                ), "image_name should have the same length as image_data"

                for img in image_data:  # [b, h, w, c]
                    Image.fromarray(img).save(
                        bytes_io, format=save_file_ext, quality=_jpg_quality
                    )
                    batched_image_bytes.append(bytes_io.getvalue())
            elif image_data.ndim == 3:
                if image_data.shape[-1] == 1:  # gray image  # [h, w, 1]
                    Image.fromarray(image_data[..., 0]).save(
                        bytes_io, format=save_file_ext, quality=_jpg_quality
                    )
                    image_data = bytes_io.getvalue()
                elif image_data.shape[-1] == 3:
                    Image.fromarray(image_data).save(
                        bytes_io, format=save_file_ext, quality=_jpg_quality
                    )
                    image_data = bytes_io.getvalue()
                else:
                    raise ValueError(
                        f"image_data shape {image_data.shape} not supported"
                    )
            elif image_data.ndim == 2:  # gray image  # [h, w]
                Image.fromarray(image_data).save(
                    bytes_io, format=save_file_ext, quality=_jpg_quality
                )
                image_data = bytes_io.getvalue()

            return image_data, batched_image_bytes

        def add_image(
            image_data: "Image.Image | np.ndarray | torch.Tensor | bytes",
            image_name: "Union[str, list[str]]",
        ):
            """add image to the zipfile

            Args:
                image_data (Image.Image | np.ndarray | torch.Tensor | bytes): can be Image.Image, np.ndarray, torch.Tensor, bytes,
                                                    shape should be [b, h, w, c], [h, w, c], [h, w, 1]
                image_name (str | list[str]): saved image names
            """

            # to bytes
            batched_image_bytes = None
            if isinstance(image_data, Image.Image):
                image_data.save(bytes_io, format=save_file_ext)
                bytes = bytes_io.getvalue()
            elif isinstance(image_data, np.ndarray):
                bytes, batched_image_bytes = to_bytes(image_data, image_name)
            elif isinstance(image_data, torch.Tensor):
                image_data = image_data.detach().cpu().numpy()
                bytes, batched_image_bytes = to_bytes(image_data, image_name)
            else:
                raise ValueError(f"image_data type {type(image_data)} not supported")

            # saving to zip file
            if batched_image_bytes is not None:
                for i, img_bytes in enumerate(batched_image_bytes):
                    zf.writestr(image_name[i], img_bytes)
            else:
                zf.writestr(image_name, bytes)

            if verbose:
                logger.info(f"add image {image_name} to zip file")

            bytes_io.seek(0)
            bytes_io.truncate()

        yield add_image

    except Exception as e:
        if verbose:
            logger.error(e, raise_error=True)
            raise e
    finally:
        if verbose:
            logger.info(f"zip file saved at {zipfile_name}, zipfile close")
        zf.close()
        bytes_io.close()


# * ==========================================================
# * some develop utilities

from warnings import warn


def deprecation_warn(message: str):
    def decorator(func):
        def wrapper(*args, **kwargs):
            warn(message)
            return func(*args, **kwargs)

        return wrapper

    return decorator
