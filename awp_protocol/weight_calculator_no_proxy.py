from dataclasses import dataclass, replace

import tensorflow as tf

@dataclass(frozen=True)
class WeightParams:
    weight_constraint: float = 5.0e-3
    step_size: float = 5.0e-4

class WeightCalculator:
    def __init__(
            self,
            classifier: tf.keras.Model,
            layers_selected_for_weight_perturbation: tuple[bool, ...] | None,
            params: WeightParams | None = None,
            **overrides
    ):
        self.step_size: tf.Tensor
        self._weight_constraint: tf.Tensor
        self._dtype = classifier.weights[0].dtype
        self._classifier = classifier
        self._perturbed_layers: tuple[bool, ...] = layers_selected_for_weight_perturbation or select_default_trained_layers_tf(self._classifier)

        self._params = params or WeightParams()
        self._params = replace(self._params, **overrides)
        self.step_size = tf.constant(self._params.step_size, dtype=self._dtype)
        self._weight_constraint = tf.constant(self._params.weight_constraint, dtype=self._dtype)

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


    def subtract_weight_perturbations(self) -> None:
        for idx in self._indices_of_selected_layers:
            self._classifier.trainable_variables[idx].assign_sub(self._weight_perturbations[idx])


    def calculate_weight_perturbations(self, gradient: list[tf.Tensor]):
        for idx in self._indices_of_selected_layers:
            if gradient[idx] is not None:
                self._weight_perturbations[idx].assign(self._calculate_perturbation_for_single_trainable_variable(gradient[idx], idx))


    def restore_model(self):
        for idx in self._indices_of_selected_layers:
            self._classifier.trainable_variables[idx].assign(self._saved_weights[idx])


    def _calculate_perturbation_for_single_trainable_variable(self, weight_gradient: tf.Tensor, idx) -> tf.Tensor:
        weight_perturbation = self._calculate_initial_perturbation(weight_gradient, idx)
        weight_perturbation = self._weight_perturbations[idx] + weight_perturbation
        return self._scale_perturbation_to_bound(weight_perturbation, idx)


    def _calculate_initial_perturbation(self, weight_gradient: tf.Tensor, idx):
        step_direction = tf.math.divide_no_nan(weight_gradient, tf.norm(weight_gradient))
        weight_perturbation = step_direction * self.step_size * self._weight_norms[idx]
        return weight_perturbation


    def _scale_perturbation_to_bound(self, weight_perturbation: tf.Tensor, idx) -> tf.Tensor:
        scale_factor = tf.math.divide_no_nan(self._weight_norms[idx], tf.norm(weight_perturbation)) * self._weight_constraint
        scale_factor = tf.minimum(tf.constant(1.0, dtype=scale_factor.dtype), scale_factor)
        return weight_perturbation * scale_factor


def _make_weight_perturbation_storage(classifier: tf.keras.models.Model, perturbed_layers: tuple[bool, ...]) -> list[tf.Variable | None]:
    return [
        tf.Variable(tf.zeros_like(variable), trainable=False) if perturbed else None
        for variable, perturbed in zip(classifier.trainable_weights, perturbed_layers)
    ]


def _make_weight_norms_storage(classifier: tf.keras.models.Model, perturbed_layers: tuple[bool, ...]) -> list[tf.Variable | None]:
    return [
        tf.Variable(tf.norm(variables), trainable=False) if perturbed else None
        for variables, perturbed in zip(classifier.trainable_variables, perturbed_layers)
    ]

def select_default_trained_layers_tf(classifier: tf.keras.Model) -> tuple[bool, ...]:
    return tuple('kernel' in variable.name for variable in classifier.trainable_variables)
