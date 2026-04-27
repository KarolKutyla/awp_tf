from abc import ABC, abstractmethod

import tensorflow as tf
from awp_protocol.losses.loss_context import LossContext


class AdversarialLoss(ABC):
    @abstractmethod
    def calculate(self, loos_context: LossContext) -> tf.Tensor:
        pass
