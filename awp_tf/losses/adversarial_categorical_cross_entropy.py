import tensorflow as tf

from awp_tf.losses.loss import AdversarialLoss
from awp_tf.losses.loss_context import LossContext


class AdversarialSparseCategoricalCrossEntropy(AdversarialLoss):
    def __init__(self):
        super().__init__()
        self._loss: tf.losses.Loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    @tf.function
    def calculate(self, x, y, x_adv, model, training: bool = False) -> tf.Tensor:
        logits_adv = model(x_adv, training=training)
        loss = self._loss(y, logits_adv)
        return loss
