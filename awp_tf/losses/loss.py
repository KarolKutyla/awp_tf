from abc import ABC, abstractmethod

import tensorflow as tf
from awp_tf.losses.loss_context import LossContext


class AdversarialLoss(ABC):
    @abstractmethod
    def calculate(self, x, y, x_adv, model, training: bool = False) -> tf.Tensor:
        pass
