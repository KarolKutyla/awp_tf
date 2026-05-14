from typing import NamedTuple

from tensorflow import Tensor

class LossContext(NamedTuple):
    x_batch: Tensor
    x_adv: Tensor
    y_batch: Tensor
    logits_clean: Tensor
    logits_adv: Tensor
