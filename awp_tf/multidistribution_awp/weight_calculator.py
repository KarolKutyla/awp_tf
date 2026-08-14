from dataclasses import dataclass, replace

import keras

from awp_tf.losses.loss_context import LossContext
from awp_tf.losses import loss, trades_loss, adversarial_categorical_cross_entropy, loss_context

import tensorflow as tf

@dataclass(frozen=True)
class WeightParams:
    weight_constraint: float = 1.0e-2
    step_size: float = 1.0e-2


class WeightCalculator:
    def __init__(
            self,
            classifier: tf.keras.Model,
            layers_selected_for_weight_perturbation: tuple[bool, ...] | None,
            params: WeightParams | None = None,
            loss: loss.AdversarialLoss = adversarial_categorical_cross_entropy.AdversarialSparseCategoricalCrossEntropy(),
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

        self._indices_of_selected_layers: tuple[int, ...] = tuple(i for i, tracked in enumerate(self._perturbed_layers) if tracked)
        self._saved_weights: tuple[tf.Variable, ...] = _initiate_memory_for_weight_perturbations(self._classifier, self._indices_of_selected_layers)
        self._weight_perturbations: tuple[tf.Variable, ...] = _initiate_memory_for_weight_perturbations(self._classifier, self._indices_of_selected_layers)
        self._weight_norms: tuple[tf.Variable, ...] = _initiate_memory_for_weight_norms(self._weight_perturbations)


    def initiate_state_for_batch_process(self) -> None:
        for i, classifier_idx in enumerate(self._indices_of_selected_layers):
            self._saved_weights[i].assign(self._classifier.trainable_variables[classifier_idx])
            self._weight_perturbations[i].assign(tf.zeros_like(self._classifier.trainable_variables[classifier_idx]))
            self._weight_norms[i].assign(tf.norm(self._classifier.trainable_variables[classifier_idx]))


    def append_weight_perturbations(self):
        for idx, perturbation in zip(self._indices_of_selected_layers, self._weight_perturbations):
            self._classifier.trainable_variables[idx].assign_add(perturbation)


    def subtract_weight_perturbations(self) -> None:
        for idx, perturbation in zip(self._indices_of_selected_layers, self._weight_perturbations):
            self._classifier.trainable_variables[idx].assign_sub(perturbation)


    def restore_model(self):
        for idx, old_value in zip(self._indices_of_selected_layers, self._saved_weights):
            self._classifier.trainable_variables[idx].assign(old_value)


    def calculate_weight_perturbation(self, x, y, x_adv, x_alt, y_alt, x_alt_adv) -> None:
        gradients = self._calculate_gradient(x, y, x_adv)
        gradients_alt = self._calculate_gradient(x_alt, y_alt, x_alt_adv)
        for gradient, gradient_alt, perturbation, norm in zip(gradients, gradients_alt, self._weight_perturbations, self._weight_norms):
            standard_step_direction = tf.math.divide_no_nan(gradient, tf.norm(gradient))
            alt_step_direction = tf.math.divide_no_nan(gradient_alt, tf.norm(gradient_alt))
            mixed_gradient = standard_step_direction + alt_step_direction
            mixed_direction = tf.math.divide_no_nan(mixed_gradient, tf.norm(mixed_gradient))
            mixed_step = mixed_direction * norm * self.step_size
            perturbation.assign(mixed_step)


    def calculate_weight_perturbation_on_subset(self, x, y, x_adv) -> None:
        gradients = self._calculate_gradient(x, y, x_adv)
        for gradient, perturbation, norm in zip(gradients, self._weight_perturbations, self._weight_norms):
            standard_step_direction = tf.math.divide_no_nan(gradient, tf.norm(gradient))
            step = standard_step_direction * norm * self.step_size
            perturbation.assign(step)


    def _calculate_gradient(self, x, y, x_adv):
        with tf.GradientTape() as tape:
            logits_clean = self._classifier(x, training=False)
            logits_adv = self._classifier(x_adv, training=False)
            ctx = LossContext(
                x_batch=x,
                x_adv=x_adv,
                y_batch=y,
                logits_clean=logits_clean,
                logits_adv=logits_adv
            )
            loss = self._loss.calculate(ctx)
        selected_variables = tuple(
            self._classifier.trainable_variables[idx]
            for idx in self._indices_of_selected_layers
        )
        return tape.gradient(loss, selected_variables, unconnected_gradients=tf.UnconnectedGradients.ZERO)



def _initiate_memory_for_weight_perturbations(classifier: keras.Model, indices_of_selected_layers: tuple[int, ...]) -> tuple[tf.Variable, ...]:
    return tuple(
        tf.Variable(
            tf.zeros_like(classifier.trainable_weights[idx]),
            trainable=False,
        )
        for idx in indices_of_selected_layers
    )


def _initiate_memory_for_weight_norms(weight_perturbations: tuple[tf.Variable, ...]) -> tuple[tf.Variable, ...]:
    return tuple(
        tf.Variable(
            tf.norm(perturbation), trainable=False
        )
        for perturbation in weight_perturbations
    )


def select_default_trained_layers_tf(classifier: tf.keras.Model) -> tuple[bool, ...]:
    return tuple('kernel' in variable.name for variable in classifier.trainable_variables)
