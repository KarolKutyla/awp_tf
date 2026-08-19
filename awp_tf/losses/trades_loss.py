import tensorflow as tf

from awp_tf.losses.loss import AdversarialLoss
from awp_tf.losses.loss_context import LossContext

class TradesLoss(AdversarialLoss):
    def __init__(self, regularization_parameter: float = 1.0):
        super().__init__()
        if regularization_parameter < 0.0:
            raise Exception(f"Beta parameter must be greater than 0. Passed value is {regularization_parameter}")
        self._regularization_parameter = tf.constant(regularization_parameter, dtype=tf.float32)
        self._mean_factor = tf.constant(regularization_parameter + 1.0, dtype=tf.float32)
        self._sparse_categorical_cross_entropy = tf.losses.SparseCategoricalCrossentropy(from_logits=True)

    # @tf.function
    def calculate(self, x, y , x_adv, model, training: bool = False) -> tf.Tensor:
        batch_size = tf.shape(x)[0]
        xx = tf.concat([x, x_adv], axis=0)
        logits = model(xx, training=training)
        logits_clean = logits[:batch_size]
        logits_adv = logits[batch_size:]

        loss_clean = self._sparse_categorical_cross_entropy(y, logits_clean)
        loss_kl = _kld_loss(logits, logits_adv)
        loss = loss_clean + loss_kl * self._regularization_parameter
        return loss / self._mean_factor

def _kld_loss(logits, logits_adv):
    p = tf.nn.softmax(logits)
    log_p = tf.nn.log_softmax(logits)
    log_q = tf.nn.log_softmax(logits_adv)
    return tf.reduce_mean(
        tf.reduce_sum(p * (log_p - log_q), axis=-1)
    )
