from abc import ABC, abstractmethod

import tensorflow as tf
from awp_tf.losses.loss_context import LossContext


class AdversarialLoss(ABC):
    @abstractmethod
    def calculate(self, x: tf.Tensor, y: tf.Tensor, x_adv: tf.Tensor, model, training: bool = False) -> tf.Tensor:
        pass

    @abstractmethod
    def calculate_attack_loss(self, x: tf.Tensor, y: tf.Tensor, x_adv: tf.Tensor, model, training: bool = False):
        pass
