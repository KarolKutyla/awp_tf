from abc import ABC, abstractmethod

import tensorflow as tf
from tensorflow import keras



class AdversarialLoss(ABC):


    @abstractmethod
    def calculate_gradient_step_loss(self, x: tf.Tensor, y: tf.Tensor, x_adv: tf.Tensor, model: keras.Model) -> tf.Tensor:
        pass


    @abstractmethod
    def calculate_weight_perturbation_loss(self, x: tf.Tensor, y: tf.Tensor, x_adv: tf.Tensor, model: keras.Model, training: bool = False) -> tf.Tensor:
        pass


    @abstractmethod
    def calculate_attack_loss(self, x: tf.Tensor, y: tf.Tensor, x_adv: tf.Tensor, model: keras.Model):
        pass


