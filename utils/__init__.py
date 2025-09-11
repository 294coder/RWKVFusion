import os
import warnings

warnings.filterwarnings("ignore", module="torch.utils")
warnings.filterwarnings("ignore", module="deepspeed.accelerator")
warnings.filterwarnings("ignore", module="torch.cuda.amp", category=FutureWarning)
warnings.filterwarnings(
    "ignore", module="kornia.feature.lightglue", category=FutureWarning
)
warnings.filterwarnings("ignore", module="lazy_loader")

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
cache_dir = os.path.join(root_dir, ".cache")

os.environ["MPLCONFIGDIR"] = os.path.join(cache_dir, "matplotlib")
os.environ["HF_HOME"] = os.path.join(cache_dir, "hf_home")

## torch opt_einsum setting
import torch
import torch.backends.opt_einsum

torch.backends.opt_einsum.strategy = "auto-hq"

## torch dynamo setting
torch.set_float32_matmul_precision("high")
# torch._dynamo.config.cache_size_limit = 64

## import utilities
from .sharpening_index._metric_legacy import *

# from .metric_sharpening import *
# from .metric_fusion import *
from .log_utils import *
from .misc import *
from .misc import dict_to_namespace as convert_config_dict
from .load_params import *
from .optim_utils import *
from .network_utils import *
from .visualize import *
from .inference_helper_func import *
from .loss_utils import *
from .save_checker import *
from .train_test_utils import *
from .progress_utils import *
from .model_perf_utils import *
from .ema_utils import *
from .deepspeed_utils import *
from .image_utils import *
from .cfg_utils import *
from .fusion_task_utils import *
from .import_utils import (
    get_module_from_obj_name,
    get_obj_by_name,
    construct_class_by_name,
    call_func_by_name,
    is_top_level_function,
    is_url,
    open_url,
)


# config load
config_load = yaml_load
