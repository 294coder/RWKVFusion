import os

import jaxtyping
from beartype import beartype
from beartype.door import is_bearable
from jaxtyping import jaxtyped
from numpy import ndarray
from torch import Tensor


class TorchTyping:
    def __init__(self, abstract_dtype):
        self.abstract_dtype = abstract_dtype

    def __getitem__(self, shapes: str):
        return self.abstract_dtype[Tensor, shapes]


class NDArrayTyping:
    def __init__(self, abstract_dtype):
        self.abstract_dtype = abstract_dtype

    def __getitem__(self, shapes: str):
        return self.abstract_dtype[ndarray, shapes]


if bool(os.getenv("type_check")):
    typing_checker = jaxtyped(typechecker=beartype)
else:
    typing_checker = jaxtyped(typechecker=None)

Float = TorchTyping(jaxtyping.Float)
Int = TorchTyping(jaxtyping.Int)
Bool = TorchTyping(jaxtyping.Bool)

FloatBatchedImage = Float["B C H W"]
FloatBatchedSeqCLast = Float["B L C"]
FloatBatchedSeqCFirst = Float["B C L"]

FloatSingleImage = Float["C H W"]
FloatSingleSeqCFirst = Float["C L"]
FloatSingleSeqCLast = Float["L C"]
