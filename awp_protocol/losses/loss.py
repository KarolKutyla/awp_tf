from typing import Optional
from abc import ABC, abstractmethod

import tensorflow as tf
from tensorflow import keras
from neural_network_analytic_tool.art_tf.losses.loss_context import LossContext


class AdversarialAttackLoss(ABC):
    @abstractmethod
    def calculate(self, loos_context: LossContext) -> tf.Tensor:
        pass
