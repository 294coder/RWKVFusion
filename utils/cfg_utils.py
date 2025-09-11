from typing import Any, Union

import hydra
from omegaconf import DictConfig, ListConfig, OmegaConf


def omegaconf_create(obj: Union[str, list, dict], parent=None, flags=None):
    """
    create an OmegaConf object from a list of args and kwargs or path.
    """
    if flags is None:
        flags = {"allow_objects": True}
    if isinstance(obj, str):
        args = OmegaConf.load(obj)
    else:
        args = OmegaConf.create(obj, parent=parent, flags=flags)

    return args


def hydra_create(cfg_name: str, init_path: str = "configs", override_args: list = []):
    hydra.initialize(config_path=init_path)
    cfg = hydra.compose(cfg_name, strict=False, overrides=override_args)

    return cfg
