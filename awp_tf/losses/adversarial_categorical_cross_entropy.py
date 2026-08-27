import keras
import tensorflow as tf

from awp_tf.losses.loss import AdversarialLoss


class AdversarialSparseCategoricalCrossEntropy(AdversarialLoss):
    def __init__(self):
        super().__init__()
        self._loss: tf.losses.Loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    # @tf.function
    def calculate(self, x: tf.Tensor, y: tf.Tensor, x_adv: tf.Tensor, model: keras.Model, training: bool = False) -> tf.Tensor:
        logits_adv = model(x_adv, training=training)
        loss = self._loss(y, logits_adv)
        return loss

    # @tf.function
    def calculate_attack_loss(self, x: tf.Tensor, y:tf.Tensor, x_adv:tf.Tensor, model: keras.Model, training: bool = False) -> tf.Tensor:
        logits_adv = model(x_adv, training=training)
        loss = self._loss(y, logits_adv)
        return loss
