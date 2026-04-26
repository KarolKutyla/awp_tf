from typing import Optional

import tensorflow as tf
from tensorflow import keras


class AWPProxyClassifier:
    def __init__(
            self,
            originator: tf.keras.Model,
            marked_trained_layers: Optional[tuple[bool]],
            step_size: float = 2/255,
            weight_constraint: float = 0.01,
    ):
        self._classifier: tf.keras.Model = _clone_classifier(originator)
        self._trained_layers: tuple[bool] = _select_perturbed_layers(originator, marked_trained_layers)
        self.step_size: float = step_size
        self._weight_constraint: float = weight_constraint
        self._weight_perturbations: list[tf.Variable] = \
            _make_weight_perturbation_storage(self._classifier)
        self._weight_norms: list[tf.Variable] = \
            _make_weight_norms_storage(self._classifier, self._trained_layers)
        self._weight_norms_multiplied_by_step_size: list[tf.Variable] = \
            _make_weight_norms_storage(self._classifier, self._trained_layers)
        self.weight_constraints: list[tf.Variable] = \
            _make_weight_constraints_storage(self._weight_norms, self._trained_layers)

    @tf.function
    def copy_originator_state(self, originator: keras.Model) -> None:
        for i, tracked in enumerate(self._trained_layers):
            self._classifier.trainable_variables[i].assign(
                originator.trainable_variables[i])
            if tracked:
                self._weight_norms[i].assign(tf.norm(originator.trainable_variables[i]))
                self.weight_constraints[i].assign(
                    self._weight_norms[i].value() * self._weight_constraint)
                self._weight_perturbations[i].assign(tf.zeros_like(self._weight_perturbations[i]))
                self._weight_norms_multiplied_by_step_size[i].assign(self._weight_norms[i].value() * self.step_size)

    @tf.function
    def subtract_perturbations_from_weights(self):
        for i, tracked in enumerate(self._trained_layers):
            if tracked:
                self._classifier.trainable_variables[i].assign_sub(self._weight_perturbations[i])

    @tf.function
    def calculate_and_apply_weight_perturbations(self, gradient: tf.Tensor, originator: tf.keras.Model) -> tf.no_op:
        for i, tracked in enumerate(self._trained_layers):
            if tracked:
                new_perturbation = self._calculate_single_weight_perturbation(gradient[i], i)
                self._weight_perturbations[i].assign(new_perturbation)
                self._classifier.trainable_variables[i].assign(
                    originator.trainable_variables[i] + new_perturbation)

    @tf.function
    def _calculate_single_weight_perturbation(self, weight_gradient: tf.Tensor, weight_index: int) -> tf.Tensor:
        gradient_norm = tf.norm(weight_gradient)
        non_zero = tf.constant(1e-6, dtype=gradient_norm.dtype)
        gradient_norm_non_zero = tf.maximum(gradient_norm, non_zero)
        calculated_weight_perturbation = weight_gradient / gradient_norm_non_zero * self._weight_norms_multiplied_by_step_size[weight_index]

        new_weight_perturbation = calculated_weight_perturbation + self._weight_perturbations[weight_index]
        scaled_weight = self._scale_single_weight_perturbation(new_weight_perturbation, weight_index)
        return scaled_weight

    @tf.function
    def _scale_single_weight_perturbation(self, weight_perturbation: tf.Tensor, weight_index: int) -> tf.Tensor:
        perturbation_norm = tf.norm(weight_perturbation)
        max_norm = self.weight_constraints[weight_index]
        eps = tf.constant(1e-6, dtype=perturbation_norm.dtype)
        scale = max_norm / tf.maximum(perturbation_norm, eps)
        scale = tf.minimum(tf.constant(1.0, dtype=scale.dtype), scale)
        return weight_perturbation * scale

    @tf.function
    def forward_pass(self, x_batch: tf.Tensor):
        return self._classifier(x_batch, training=True)

    @tf.function
    def get_trainable_variables(self):
        return self._classifier.trainable_variables


def _clone_classifier(originator: tf.keras.Model) -> tf.keras.Model:
    proxy_classifier = tf.keras.models.clone_model(originator)
    proxy_classifier.set_weights(originator.get_weights())
    return proxy_classifier


def _select_perturbed_layers(model: keras.Model, trained_layers: Optional[tuple[bool]]) -> tuple[bool]:
    if trained_layers is None:
        return tuple('kernel' in variable.name for variable in model.trainable_variables)
    else:
        return trained_layers


def _make_weight_perturbation_storage(classifier: keras.models.Model) -> list[tf.Variable]:
    return [tf.Variable(tf.zeros_like(variable), trainable=False) for variable in classifier.trainable_weights]


def _make_weight_norms_storage(classifier: keras.models.Model, trained_layers: tuple[bool]) -> list[tf.Variable]:
    return [tf.Variable(tf.norm(variables)) if tracked
            else None for variables, tracked
            in zip(classifier.trainable_variables, trained_layers)]


def _make_weight_constraints_storage(weight_norms: list[tf.Variable], trained_layers: tuple[bool]) -> list[tf.Variable]:
    return [
        tf.Variable(weight_size) if tracked else None
        for weight_size, tracked in zip(weight_norms, trained_layers)
    ]
