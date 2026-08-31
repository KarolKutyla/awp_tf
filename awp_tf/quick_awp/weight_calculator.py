from dataclasses import dataclass, replace

import keras

from awp_tf.losses.loss_context import LossContext
from awp_tf.losses.loss import AdversarialLoss
from awp_tf.losses.adversarial_categorical_cross_entropy import AdversarialSparseCategoricalCrossEntropy

import tensorflow as tf

@dataclass(frozen=True)
class WeightParams:
    weight_constraint: float = 1.0e-2


class WeightCalculator:
    def __init__(
            self,
            classifier: tf.keras.Model,
            layers_selected_for_weight_perturbation: tuple[bool | float, ...] | None,
            params: WeightParams | None = None,
            loss: AdversarialLoss = AdversarialSparseCategoricalCrossEntropy(),
            **overrides
    ):
        self.step_size: tf.Tensor
        self._weight_constraint: tf.Tensor
        self._dtype = classifier.weights[0].dtype
        self._classifier = classifier
        self._loss = loss

        self._params = params or WeightParams()
        self._params = replace(self._params, **overrides)
        self._weight_constraint = self._params.weight_constraint

        self._perturbation_scales = _normalize_layer_scales(classifier, layers_selected_for_weight_perturbation)
        self._indices_of_selected_layers: tuple[int, ...] = tuple(i for i, value in enumerate(self._perturbation_scales) if value != 0.0)
        self._saved_weights: tuple[tf.Variable, ...] = _initiate_memory_for_weight_perturbations(self._classifier, self._indices_of_selected_layers)
        self._weight_perturbations: tuple[tf.Variable, ...] = _initiate_memory_for_weight_perturbations(self._classifier, self._indices_of_selected_layers)
        self._weight_norms: tuple[tf.Variable, ...] = _initiate_memory_for_weight_norms(self._weight_perturbations)


    def initiate_state_for_batch_process(self) -> None:
        for i, classifier_idx in enumerate(self._indices_of_selected_layers):
            self._saved_weights[i].assign(self._classifier.trainable_variables[classifier_idx])
            self._weight_perturbations[i].assign(tf.zeros_like(self._classifier.trainable_variables[classifier_idx]))
            self._weight_norms[i].assign(tf.norm(self._classifier.trainable_variables[classifier_idx]))


    def apply_weight_perturbations(self):
        for idx, perturbation, original_weight in zip(self._indices_of_selected_layers, self._weight_perturbations, self._saved_weights):
            self._classifier.trainable_variables[idx].assign(original_weight.value() + perturbation.value())


    def subtract_weight_perturbations(self) -> None:
        for idx, perturbation in zip(self._indices_of_selected_layers, self._weight_perturbations):
            self._classifier.trainable_variables[idx].assign_sub(perturbation)


    def restore_model(self):
        for idx, old_value in zip(self._indices_of_selected_layers, self._saved_weights):
            self._classifier.trainable_variables[idx].assign(old_value)


    def calculate_weight_perturbation(self, x: tf.Tensor, y: tf.Tensor, x_adv: tf.Tensor) -> None:
        gradients = self._calculate_gradient(x, y, x_adv)
        for i, (gradient, perturbation, norm) in enumerate(zip(gradients, self._weight_perturbations, self._weight_norms)):
            step_direction = tf.math.divide_no_nan(gradient, tf.norm(gradient))
            step = step_direction * norm * self._perturbation_scales[i] * self._weight_constraint
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
            loss = self._loss.calculate_weight_perturbation_loss(ctx)
        selected_variables = tuple(
            self._classifier.trainable_variables[idx]
            for idx in self._indices_of_selected_layers
        )
        return tape.gradient(loss, selected_variables, unconnected_gradients=tf.UnconnectedGradients.ZERO)

def _normalize_layer_scales(
        classifier: tf.keras.Model,
        scales: tuple[bool | float, ...] | None
) -> tuple[float, ...]:

    if scales is None:
        return tuple(
            1.0 if "kernel" in v.name else 0.0
            for v in classifier.trainable_variables
        )

    return tuple(
        1.0 if x is True else
        0.0 if x is False else
        float(x)
        for x in scales
    )

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
