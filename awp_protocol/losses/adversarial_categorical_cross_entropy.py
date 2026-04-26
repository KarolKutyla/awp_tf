import tensorflow as tf

from loss import AdversarialAttackLoss

from neural_network_analytic_tool.art_tf.losses.loss_context import LossContext


class AdversarialCategoricalCrossEntropy(AdversarialAttackLoss):
    def __init__(self):
        super().__init__()
        self._categorical_cross_entropy = tf.losses.CategoricalCrossentropy(from_logits=True)

    @tf.function
    def calculate(self, loss_context: LossContext) -> tf.Tensor:
        y = loss_context.y
        logits_adv = loss_context.logits_adv
        loss = self._categorical_cross_entropy(y, logits_adv)
        return loss
