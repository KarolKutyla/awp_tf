from typing import Optional
from dataclasses import dataclass, replace

import tensorflow as tf
from tensorflow import keras

@dataclass(frozen=True)
class AWPProxyParams:
    step_size: float = 2/255
    weight_constraint: float = 0.01

class AWPProxyCalculations:
    def __init__(
            self,
            bound_classifier: tf.keras.Model,
            marked_trained_layers: tuple[bool, ...],
            params: AWPProxyParams| None = None,
            **overrides
    ):
        self._bound_classifier: tf.keras.Model = bound_classifier
        self._trained_layers: tuple[bool, ...] = marked_trained_layers

        self._params = params or AWPProxyParams()
        self._params = replace(self._params, **overrides)
        self.step_size: float = self._params.step_size
        self._weight_constraint: float = self._params.weight_constraint

        self._weight_perturbations: list[tf.Variable] = \
            _make_weight_perturbation_storage(self._bound_classifier)
        self._weight_norms: list[tf.Variable | None] = \
            _make_weight_norms_storage(self._bound_classifier, self._trained_layers)
        self._weight_norms_multiplied_by_step_size: list[tf.Variable | None] = \
            _make_weight_norms_storage(self._bound_classifier, self._trained_layers)
        self.weight_constraints: list[tf.Variable | None] = \
            _make_weight_constraints_storage(self._weight_norms, self._trained_layers)

    @property
    def trainable_variables(self):
        return self._bound_classifier.trainable_variables

    # @tf.function
    def copy_originator_state(self, originator: keras.Model) -> None:
        for i, tracked in enumerate(self._trained_layers):
            self._bound_classifier.trainable_variables[i].assign(
                originator.trainable_variables[i])
            if tracked:
                self._weight_norms[i].assign(tf.norm(originator.trainable_variables[i]))
                self.weight_constraints[i].assign(
                    self._weight_norms[i].value() * self._weight_constraint)
                self._weight_perturbations[i].assign(tf.zeros_like(self._weight_perturbations[i]))
                self._weight_norms_multiplied_by_step_size[i].assign(self._weight_norms[i].value() * self.step_size)

    # @tf.function
    def subtract_perturbations_from_weights(self):
        for i, tracked in enumerate(self._trained_layers):
            if tracked:
                self._bound_classifier.trainable_variables[i].assign_sub(self._weight_perturbations[i])

    # @tf.function
    def calculate_and_store_weight_perturbation(self, gradient: list[tf.Tensor]) -> None:
        for i, tracked in enumerate(self._trained_layers):
            if tracked:
                if gradient[i] is not None:
                    new_perturbation = self._calculate_single_weight_perturbation(gradient[i], i)
                    self._weight_perturbations[i].assign(new_perturbation)

    # @tf.function
    def apply_stored_weight_perturbation(self, originator: keras.Model) -> None:
        for i, tracked in enumerate(self._trained_layers):
            if tracked:
                self._bound_classifier.trainable_variables[i].assign(
                    originator.trainable_variables[i] + self._weight_perturbations[i])

    # @tf.function
    def _calculate_single_weight_perturbation(self, weight_gradient: tf.Tensor, weight_index: int) -> tf.Tensor:
        gradient_norm = tf.norm(weight_gradient)
        non_zero = tf.constant(1e-6, dtype=gradient_norm.dtype)
        gradient_norm_non_zero = tf.maximum(gradient_norm, non_zero)
        calculated_weight_perturbation = weight_gradient / gradient_norm_non_zero * self._weight_norms_multiplied_by_step_size[weight_index]

        new_weight_perturbation = calculated_weight_perturbation + self._weight_perturbations[weight_index]
        scaled_weight = self._scale_single_weight_perturbation(new_weight_perturbation, weight_index)
        return scaled_weight

    # @tf.function
    def _scale_single_weight_perturbation(self, weight_perturbation: tf.Tensor, weight_index: int) -> tf.Tensor:
        perturbation_norm = tf.norm(weight_perturbation)
        max_norm = self.weight_constraints[weight_index]
        eps = tf.constant(1e-6, dtype=perturbation_norm.dtype)
        scale = max_norm / tf.maximum(perturbation_norm, eps)
        scale = tf.minimum(tf.constant(1.0, dtype=scale.dtype), scale)
        return weight_perturbation * scale

    # @tf.function
    def forward_pass(self, x_batch: tf.Tensor):
        return self._bound_classifier(x_batch, training=False)

def _make_weight_perturbation_storage(classifier: keras.models.Model) -> list[tf.Variable]:
    return [tf.Variable(tf.zeros_like(variable), trainable=False) for variable in classifier.trainable_weights]


def _make_weight_norms_storage(classifier: keras.models.Model, trained_layers: tuple[bool, ...]) -> list[tf.Variable | None]:
    return [tf.Variable(tf.norm(variables)) if tracked
            else None for variables, tracked
            in zip(classifier.trainable_variables, trained_layers)]


def _make_weight_constraints_storage(weight_norms: list[tf.Variable | None], trained_layers: tuple[bool, ...]) -> list[tf.Variable | None]:
    return [
        tf.Variable(weight_size) if tracked else None
        for weight_size, tracked in zip(weight_norms, trained_layers)
    ]
