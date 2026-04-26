from tensorflow import Tensor


class LossContext:
    def __init__(
            self,
            x: Tensor,
            x_pert: Tensor,
            logits: Tensor,
            logits_pert: Tensor,
            y: Tensor
    ):
        self.x = x
        self.x_adv = x_pert,
        self.logits = logits,
        self.logits_adv = logits_pert
        self.y = y