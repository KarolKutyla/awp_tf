from dataclasses import dataclass, replace

import tensorflow as tf
from tensorflow import keras

from awp_tf.api.awp_params import AWPParams


class Calculator:
    def __init__(
            self,
            classifier: tf.keras.Model,
            layer_scales: tuple[float, ...],
            params: AWPParams
    ):
        self._data_dtype = classifier.weights[0].dtype
        self._step_size: tf.Tensor = tf.cast(params.step_size, dtype=self._data_dtype)
        self._perturbation_size_constraint: tf.Tensor = tf.cast(params.perturbation_size_constraint, dtype=self._data_dtype)
        self._alternate_distribution_tradeoff: tf.Tensor = tf.cast(params.alternate_distribution_tradeoff, dtype=self._data_dtype)
        self._learning_rate = classifier.optimizer.learning_rate
        # self._classifier = classifier

        self._layer_scales = layer_scales
        self._applied_layers: tuple[int, ...] = tuple(i for i, value in enumerate(self._layer_scales) if value != 0.0)
        self._saved_weights: tuple[tf.Variable, ...] = _initiate_memory_for_weight_perturbations(classifier, self._applied_layers)
        self._weight_perturbations: tuple[tf.Variable, ...] = _initiate_memory_for_weight_perturbations(classifier, self._applied_layers)
        self._weight_norms: tuple[tf.Variable, ...] = _initiate_memory_for_weight_norms(self._weight_perturbations)


    def initiate_state_for_batch_process(self, classifier: keras.Model) -> None:
        for i, classifier_idx in enumerate(self._applied_layers):
            self._saved_weights[i].assign(classifier.trainable_variables[classifier_idx])
            self._weight_perturbations[i].assign(tf.zeros_like(classifier.trainable_variables[classifier_idx]))
            self._weight_norms[i].assign(tf.norm(classifier.trainable_variables[classifier_idx]))


    def apply_weight_perturbations(self, classifier: keras.Model):
        for idx, perturbation, original_weight in zip(self._applied_layers, self._weight_perturbations, self._saved_weights):
            classifier.trainable_variables[idx].assign(original_weight.value() + perturbation.value())


    def restore_original_weights(self, classifier: keras.Model) -> None:
        for idx, old_value in zip(self._applied_layers, self._saved_weights):
            classifier.trainable_variables[idx].assign(old_value)


    def calculate_weight_perturbation(self, gradients: tuple[tf.Tensor, ...]) -> None:
        for idx, gradient, perturbation, norm in zip(self._applied_layers, gradients, self._weight_perturbations, self._weight_norms):
            step_direction = tf.math.divide_no_nan(gradient, tf.norm(gradient))
            step = step_direction * norm * self._layer_scales[idx] * self._step_size
            perturbation.assign(step)


    def calculate_completely_random_perturbation(self, gradients: tuple[tf.Tensor, ...]) -> None:
        for idx, gradient, perturbation, norm in zip(self._applied_layers, gradients, self._weight_perturbations, self._weight_norms):
            random_perturbation = tf.random.normal(shape=gradient.shape)
            random_direction = tf.math.divide_no_nan(random_perturbation, tf.norm(random_perturbation))
            step = random_direction * norm * self._layer_scales[idx] * self._step_size
            perturbation.assign(step)

    def calculate_random_perturbation_that_match_gradient_sign(self, gradients: tuple[tf.Tensor, ...]) -> None:
        for idx, gradient, perturbation, norm in zip(self._applied_layers, gradients, self._weight_perturbations, self._weight_norms):
            random_perturbation = tf.random.normal(shape=gradient.shape)
            random_perturbation_correct_direction = tf.abs(random_perturbation) * tf.sign(gradient)
            random_direction = tf.math.divide_no_nan(random_perturbation_correct_direction, tf.norm(random_perturbation_correct_direction))
            step = random_direction * norm * self._layer_scales[idx] * self._step_size
            perturbation.assign(step)

    def calculate_random_perturbation_for_smooth_params(self, gradients: tuple[tf.Tensor, ...]) -> None:
        for idx, gradient, perturbation, norm in zip(
                self._applied_layers, gradients, self._weight_perturbations, self._weight_norms
        ):
            gradient_norm = tf.abs(gradient)
            gradient_norm_avg = tf.reduce_mean(gradient_norm)
            smooth_params_mask = tf.cast(
                gradient_norm <= gradient_norm_avg,
                gradient.dtype,
            )

            random_values = tf.random.normal(shape=gradient.shape)
            random_values_matching_gradient_signs = tf.abs(random_values) * tf.sign(gradient)

            random_smooth_values = random_values_matching_gradient_signs * smooth_params_mask
            perturbation_direction = tf.math.divide_no_nan(random_smooth_values, tf.norm(random_smooth_values))
            step = perturbation_direction * norm * self._layer_scales[idx] * self._step_size

            perturbation.assign(step)


    def calculate_random_perturbation_for_steep_params(self, gradients: tuple[tf.Tensor, ...]) -> None:
        for idx, gradient, perturbation, norm in zip(
                self._applied_layers, gradients, self._weight_perturbations, self._weight_norms
        ):
            gradient_norm = tf.abs(gradient)
            gradient_norm_avg = tf.reduce_mean(gradient_norm)
            steep_params_mask = tf.cast(
                gradient_norm > gradient_norm_avg,
                gradient.dtype,
            )

            random_values = tf.random.normal(shape=gradient.shape)
            random_values_matching_gradient_signs = tf.abs(random_values) * tf.sign(gradient)

            random_steep_values = random_values_matching_gradient_signs * steep_params_mask
            perturbation_direction = tf.math.divide_no_nan(random_steep_values, tf.norm(random_steep_values))
            step = perturbation_direction * norm * self._layer_scales[idx] * self._step_size

            perturbation.assign(step)


    def get_applied_layers_indices(self):
        return self._applied_layers


    def calculate_random_multi_batch_perturbation_for_single_steep_params(self, gradients: tuple[tf.Tensor, ...]) -> None:
        ...


    def calculate_random_multi_batch_perturbation_for_double_steep_params(self, gradients: tuple[tf.Tensor, ...]) -> None:
        ...



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
