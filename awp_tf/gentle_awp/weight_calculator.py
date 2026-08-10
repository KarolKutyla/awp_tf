from dataclasses import dataclass, replace

from awp_tf.losses.loss_context import LossContext
from awp_tf.losses import loss, trades_loss, adversarial_categorical_cross_entropy, loss_context

import tensorflow as tf

@dataclass(frozen=True)
class WeightParams:
    awp_steps: int = 1
    step_size: float = 5.0e-4

class WeightCalculator:
    def __init__(
            self,
            classifier: tf.keras.Model,
            layers_selected_for_weight_perturbation: tuple[bool, ...] | None,
            params: WeightParams | None = None,
            loss: loss.AdversarialLoss = adversarial_categorical_cross_entropy.AdversarialSparseCategoricalCrossEntropy()
            **overrides
    ):
        self.step_size: tf.Tensor
        self._weight_constraint: tf.Tensor
        self._dtype = classifier.weights[0].dtype
        self._classifier = classifier
        self._perturbed_layers: tuple[bool, ...] = layers_selected_for_weight_perturbation or select_default_trained_layers_tf(self._classifier)
        self._loss = loss

        self._params = params or WeightParams()
        self._params = replace(self._params, **overrides)
        self.step_size = tf.constant(self._params.step_size, dtype=self._dtype)
        self._awp_steps = tf.constant(self._params.awp_steps, dtype=tf.int32)


        self._indices_of_selected_layers = [i for i, tracked in enumerate(self._perturbed_layers) if tracked]
        self._saved_weights: list[tf.Variable | None] = _make_weight_perturbation_storage(self._classifier, self._perturbed_layers)
        self._weight_perturbations: list[tf.Variable | None] = _make_weight_perturbation_storage(self._classifier, self._perturbed_layers)
        self._weight_norms: list[tf.Variable | None] = _make_weight_norms_storage(self._classifier, self._perturbed_layers)


    def reset_weight_perturbations(self) -> None:
        for idx in self._indices_of_selected_layers:
            self._saved_weights[idx].assign(self._classifier.trainable_variables[idx])
            self._weight_perturbations[idx].assign(tf.zeros_like(self._classifier.trainable_variables[idx]))
            self._weight_norms[idx].assign(tf.norm(self._classifier.trainable_variables[idx]))


    def apply_weight_perturbations(self):
        for idx in self._indices_of_selected_layers:
            self._classifier.trainable_variables[idx].assign(self._saved_weights[idx] + self._weight_perturbations[idx])


    def restore_model(self):
        for idx in self._indices_of_selected_layers:
            self._classifier.trainable_variables[idx].assign(self._saved_weights[idx])


    def subtract_weight_perturbations(self) -> None:
        for idx in self._indices_of_selected_layers:
            self._classifier.trainable_variables[idx].assign_sub(self._weight_perturbations[idx])


    def calculate_weight_perturbation(self, ctx: LossContext) -> None:
        i0 = tf.constant(0, dtype=tf.int32)

        def cond(i, ctx):
            return i < self._awp_steps

        _, _, _ = tf.nest.map_structure(
            tf.stop_gradient,
            tf.while_loop(cond, self._calculate_weight_perturbation_body, [i0, ctx], parallel_iterations=1))


    def _calculate_weight_perturbation_body(self, i, ctx: LossContext):
        with tf.GradientTape() as tape:
            ctx = ctx._replace(logits_clean=self._classifier(ctx.x_batch), logits_adv=self._classifier(ctx.x_adv))
            loss = self._loss.calculate(ctx)
        gradient = tape.gradient(loss, self._classifier.trainable_variables)

        for idx in self._indices_of_selected_layers:
            if gradient[idx] is not None:
                self._weight_perturbations[idx].assign(
                    self._calculate_perturbation_for_single_trainable_variable(gradient[idx], idx))

        self.apply_weight_perturbations()
        return i + 1, x, y


    def _calculate_perturbation_for_single_trainable_variable(self, weight_gradient: tf.Tensor, idx) -> tf.Tensor:
        step_sign = tf.math.sign(weight_gradient)
        weight_perturbation = (step_sign * self._saved_weights[idx]) * self.step_size
        return weight_perturbation



def _make_weight_perturbation_storage(classifier: tf.keras.models.Model, perturbed_layers: tuple[bool, ...]) -> list[tf.Variable | None]:
    return [tf.Variable(tf.zeros_like(variable), trainable=False) if perturbed else None
            for variable, perturbed in zip(classifier.trainable_weights, perturbed_layers)]


def _make_weight_norms_storage(classifier: tf.keras.models.Model, perturbed_layers: tuple[bool, ...]) -> list[tf.Variable | None]:
    return [tf.Variable(tf.norm(variables), trainable=False) if perturbed else None
            for variables, perturbed in zip(classifier.trainable_variables, perturbed_layers)]


def select_default_trained_layers_tf(classifier: tf.keras.Model) -> tuple[bool, ...]:
    return tuple('kernel' in variable.name for variable in classifier.trainable_variables)
