import keras
import tensorflow as tf

from awp_tf.losses.loss import AdversarialLoss



class AdversarialSparseCategoricalCrossEntropy(AdversarialLoss):


    def __init__(self):
        super().__init__()
        self._loss: tf.losses.Loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)


    def calculate_gradient_step_loss(self, x: tf.Tensor, y: tf.Tensor, x_adv: tf.Tensor, model: keras.Model) -> tf.Tensor:
        logits_adv = model(x_adv, training=True)
        loss = self._loss(y, logits_adv)
        return loss


    def calculate_weight_perturbation_loss(self, x: tf.Tensor, y: tf.Tensor, x_adv: tf.Tensor, model: keras.Model, training: bool = False) -> tf.Tensor:
        logits_adv = model(x_adv, training=False)
        loss = self._loss(y, logits_adv)
        return loss


    def calculate_attack_loss(self, x: tf.Tensor, y:tf.Tensor, x_adv:tf.Tensor, model: keras.Model) -> tf.Tensor:
        logits_adv = model(x_adv, training=False)
        loss = self._loss(y, logits_adv)
        return loss
