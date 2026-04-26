import tensorflow as tf
from tensorflow import keras

from loss import AdversarialAttackLoss

from neural_network_analytic_tool.art_tf.losses.loss_context import LossContext


class TradesLoss(AdversarialAttackLoss):
    def __init__(self, beta: float = 2.0, eps: float = 1e-3):
        super().__init__()
        self._value_instead_of_zero = eps
        if beta <= 0:
            raise Exception(f"Beta parameter must be greater than 0. Passed value is {beta}")
        self._regularization_parameter = beta
        self._categorical_cross_entropy = tf.losses.CategoricalCrossentropy(from_logits=True)
        self._kl_divergence = keras.losses.KLDivergence(reduction="sum_over_batch_size")

    @tf.function
    def calculate(self, loss_context: LossContext) -> tf.Tensor:
        y = loss_context.y
        logits = loss_context.logits
        logits_adv = loss_context.logits_adv

        min_boundry = self._value_instead_of_zero
        max_boundry = tf.dtypes.as_dtype(logits.dtype).max
        logits_clipped = tf.clip_by_value(logits, clip_value_min=min_boundry, clip_value_max=max_boundry)

        loss_clean = self._categorical_cross_entropy(y, logits)
        loss_kl = self._kl_divergence(logits_clipped, logits_adv)
        loss = loss_clean + self._regularization_parameter * loss_kl
        return loss
