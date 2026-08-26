import tensorflow as tf
from tensorflow import keras

from awp_tf.api.awp_params import AWPParams
from awp_tf.losses.loss import AdversarialLoss
from awp_tf.attacks.attack import EvasionAttack
from awp_tf.api.awp_operations import Calculator


class AWP:
    def __init__(
            self,
            classifier: keras.Model,
            robust_loss:AdversarialLoss,
            attack: EvasionAttack,
            used_layers: tuple[float, ...],
            awp_params: AWPParams = AWPParams()
    ):
        self._classifier = classifier
        self._robust_loss = robust_loss
        self._attack = attack
        self._calculator = Calculator(self._classifier, used_layers, awp_params)

        self._alternate_iteration = awp_params.alternate_iterations
        self._awp_steps = awp_params.steps


    @tf.function(jit_compile=True)
    def process_batch(
            self,
            x_batch: tf.Tensor,
            y_batch: tf.Tensor
    ) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:

        original_x_batch_adv = self._attack.generate(x_batch, y_batch)
        validation_data = self._validation_metrics(x_batch, y_batch, original_x_batch_adv)

        self._calculator.initiate_state_for_batch_process(self._classifier)
        x_batch_adv = original_x_batch_adv
        for iteration in range(self._alternate_iteration):
            if iteration > 0:
                x_batch_adv = self._attack.generate(x_batch, y_batch)
            for step in range(self._awp_steps):
                gradients = self._calculate_gradient_for_perturbation(x_batch, y_batch, x_batch_adv)
                self._calculator.calculate_weight_perturbation(gradients)
                self._calculator.apply_weight_perturbations(self._classifier)

        gradient = self._calculate_gradient_for_update(x_batch, y_batch, original_x_batch_adv)
        self._calculator.restore_original_weights(self._classifier)
        self._classifier.optimizer.apply(gradient)

        return validation_data


    def _calculate_gradient_for_perturbation(self, x, y, x_adv):
        selected_variables = tuple(
            self._classifier.trainable_variables[idx]
            for idx in self._calculator.get_applied_layers_indices()
        )
        with tf.GradientTape() as tape:
            loss = self._robust_loss.calculate(x, y, x_adv, self._classifier, training=False)
        return tape.gradient(loss, selected_variables, unconnected_gradients=tf.UnconnectedGradients.ZERO)


    def _calculate_gradient_for_update(self, x, y, x_adv):
        with tf.GradientTape() as tape:
            robust_loss = self._robust_loss.calculate(x, y, x_adv, self._classifier, training=True)
        return tape.gradient(robust_loss, self._classifier.trainable_variables)


    def _validation_metrics(self, x, y, x_adv) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        logits_clean = self._classifier(x, training=False)
        loss_on_clean_examples = self._classifier.loss(y_true=y, y_pred=logits_clean)
        logits_adv = self._classifier(x_adv, training=False)
        loss_on_adversarial_examples = self._classifier.loss(y_true=y, y_pred=logits_adv)
        return loss_on_clean_examples, logits_clean, loss_on_adversarial_examples, logits_adv
