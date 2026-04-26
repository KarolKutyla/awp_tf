from abc import ABC, abstractmethod

import tensorflow as tf
from tensorflow import keras


class TensorflowEvasionAttack(ABC):

    def __init__(self):
        ...

    @abstractmethod
    def generate_attack(self, x_batch: tf.Tensor, y_batch: tf.Tensor) -> tf.Tensor:
        ...
