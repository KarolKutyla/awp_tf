import tensorflow as tf
from tensorflow import keras

from awp_protocol.losses.loss import AdversarialLoss
from awp_protocol.losses.loss_context import LossContext

class TradesLoss(AdversarialLoss):
    def __init__(self, beta: float = 2.0, eps: float = 1e-3):
        super().__init__()
        self._value_instead_of_zero = eps
        if beta <= 0:
            raise Exception(f"Beta parameter must be greater than 0. Passed value is {beta}")
        self._regularization_parameter = beta
        self._sparse_categorical_cross_entropy = tf.losses.SparseCategoricalCrossentropy(from_logits=True)
        self._kl_divergence = keras.losses.KLDivergence(reduction="sum_over_batch_size")

    @tf.function
    def calculate(self, loss_context: LossContext) -> tf.Tensor:
        y = loss_context.y_true
        logits = loss_context.logits_out
        logits_adv = loss_context.logits_pert

        min_boundry = self._value_instead_of_zero
        max_boundry = tf.dtypes.as_dtype(logits.dtype).max
        logits_clipped = tf.clip_by_value(logits, clip_value_min=min_boundry, clip_value_max=max_boundry)

        loss_clean = self._sparse_categorical_cross_entropy(y, logits)
        loss_kl = self._kl_divergence(logits_clipped, logits_adv)
        loss = loss_clean + self._regularization_parameter * loss_kl
        tf.print("gradient calculated", loss)
        return loss
