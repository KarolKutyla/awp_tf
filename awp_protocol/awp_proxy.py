from dataclasses import dataclass, replace

import tensorflow as tf
from numpy.ma.core import indices
from tensorflow import keras
from tensorflow import stack

@dataclass(frozen=True)
class AWPProxyParams:
    weight_constraint: float = 5.0e-3
    step_size: float = weight_constraint / 50

class AWPProxyCalculations:
    def __init__(
            self,
            originator: tf.keras.Model,
            bound_classifier: tf.keras.Model,
            layers_selected_for_weight_perturbation: tuple[bool, ...],
            params: AWPProxyParams| None = None,
            **overrides
    ):
        self._dtype = tf.float32

        self._originator = originator
        self._bound_classifier: tf.keras.Model = bound_classifier


        self._params = params or AWPProxyParams()
        self._params = replace(self._params, **overrides)
        self.step_size: tf.Tensor = tf.constant(self._params.step_size, dtype=self._dtype)
        self._weight_constraint: tf.Tensor = tf.constant(self._params.weight_constraint, dtype=self._dtype)

        self._trained_layers = layers_selected_for_weight_perturbation
        self._active_indices = [i for i, tracked in enumerate(layers_selected_for_weight_perturbation) if tracked]

        self._weight_perturbations: list[tf.Variable] = \
            _make_weight_perturbation_storage(self._bound_classifier)
        self._weight_norms: list[tf.Variable | None] = \
            _make_weight_norms_storage(self._bound_classifier, self._trained_layers)


    @property
    def trainable_variables(self):
        return self._bound_classifier.trainable_variables

    # @tf.function
    def batch_process_begin(self) -> None:
        for i, weight in enumerate(self._originator.trainable_variables):
            self._bound_classifier.trainable_variables[i].assign(weight)
        for i in self._active_indices:
            self._weight_norms[i].assign(tf.norm(self._bound_classifier.trainable_variables[i]))
            self._weight_perturbations[i].assign(tf.zeros_like(self._weight_perturbations[i]))

    # @tf.function
    def subtract_perturbations_from_weights(self):
        for i in self._active_indices:
            self._bound_classifier.trainable_variables[i].assign_sub(self._weight_perturbations[i])

    # @tf.function
    def calculate_and_update_weight_perturbation(self, gradient: list[tf.Tensor]) -> None:
        for idx in self._active_indices:
            if gradient[idx] is not None:
                new_perturbation = self._calculate_single_weight_perturbation(gradient[idx], idx)
                self._weight_perturbations[idx].assign(new_perturbation)
                self._bound_classifier.trainable_variables[idx].assign(
                    self._originator.trainable_variables[idx] + self._weight_perturbations[idx])

    # # @tf.function
    # def apply_stored_weight_perturbation(self) -> None:
    #     for i, tracked in enumerate(self._trained_layers):
    #         if tracked:


    # @tf.function
    def _calculate_single_weight_perturbation(self, weight_gradient: tf.Tensor, idx) -> tf.Tensor:
        initial_weight_perturbation = self._calculate_initial_weight_perturbation_from_gradient(weight_gradient, idx)
        weight_perturbation = self._weight_perturbations[idx] + initial_weight_perturbation
        projected_weight_perturbation = self._project_single_weight_perturbation(weight_perturbation, idx)
        return projected_weight_perturbation

    def _calculate_initial_weight_perturbation_from_gradient(self, weight_gradient: tf.Tensor, idx):
        gradient_norm = tf.norm(weight_gradient)
        normalized_gradient = tf.math.divide_no_nan(weight_gradient, gradient_norm)
        weight_perturbation = self.step_size * normalized_gradient * self._weight_norms[idx]
        return weight_perturbation

    # @tf.function
    def _project_single_weight_perturbation(self, weight_perturbation: tf.Tensor, idx) -> tf.Tensor:
        perturbation_norm = tf.norm(weight_perturbation)
        scale_factor = tf.math.divide_no_nan(self._weight_norms[idx], perturbation_norm) * self._weight_constraint
        scale_factor = tf.minimum(tf.constant(1.0, dtype=scale_factor.dtype), scale_factor)
        return weight_perturbation * scale_factor

    # @tf.function
    def forward_pass(self, x_batch: tf.Tensor, training: bool = True):
        return self._bound_classifier(x_batch, training=training)

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
