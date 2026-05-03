from abc import ABC, abstractmethod

import tensorflow as tf

class TensorflowEvasionAttack(ABC):

    def __init__(self, model: tf.keras.Model):
        self.model = model

    @abstractmethod
    def generate(self, x_batch: tf.Tensor, y_batch: tf.Tensor) -> tf.Tensor:
        ...
