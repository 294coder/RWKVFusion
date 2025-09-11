# ruff: noqa
import torch
from accelerate.utils.other import TORCH_SAFE_GLOBALS
from collections import defaultdict

# ============================== Optimizers ==============================
from .adam_mini import Adam_mini
from .ada_ema_mix import AdEMAMix
from .muon_naive import Muon
from .SOAP import SOAP  # original implementation
from .cautious_adamw import AdamW as CautiousAdamW
from .cautious_lion import Lion as CautiousLion
from .mars_adamw import MARS

import heavyball.utils
heavyball.utils.compile_mode = None

from heavyball import (
    # SOAP family ===================
    ForeachSOAP,
    PaLMForeachSOAP,
    PrecondScheduleForeachSOAP,
    PrecondSchedulePaLMForeachSOAP,
    # PSGD family ===================
    ForeachPSGDKron,
    ForeachPurePSGD,
    ForeachDelayedPSGD,
    utils as heavyball_utils,
)
# add to make torch serialization safe
_SAFE_OPTIMIZERS_CLS = [
    SOAP,
    ForeachSOAP,
    PaLMForeachSOAP,
    PrecondScheduleForeachSOAP,
    PrecondSchedulePaLMForeachSOAP,
    ForeachPSGDKron,
    ForeachPurePSGD,
    ForeachDelayedPSGD,
    Muon,
    Adam_mini,
    AdEMAMix,
    CautiousAdamW,
    CautiousLion,
    MARS,
]
torch.serialization.add_safe_globals(_SAFE_OPTIMIZERS_CLS)
TORCH_SAFE_GLOBALS.extend(_SAFE_OPTIMIZERS_CLS)
TORCH_SAFE_GLOBALS.extend([defaultdict, dict])

# ============================== Models ==============================

# Unet used in EMMA loss for image fusion
from .unet5 import UNet5 as TranslationUnet
