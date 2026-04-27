from typing import NamedTuple

from tensorflow import Tensor

class LossContext(NamedTuple):
    x_batch: Tensor
    x_pert: Tensor
    y_true: Tensor
    logits_out: Tensor
    logits_pert: Tensor
