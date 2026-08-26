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


    def calculate_multi_batch_weight_perturbation(self, gradients: tuple[tf.Tensor, ...], alternative_distribution_gradients: tuple[tf.Tensor, ...]) -> None:
        for idx, gradient, gradient_alt, perturbation, norm in zip(
                self._applied_layers, gradients, alternative_distribution_gradients, self._weight_perturbations, self._weight_norms
        ):
            compared_gradients = tf.math.divide_no_nan(
                tf.abs(gradient_alt) - tf.abs(gradient),
                tf.abs(gradient) + tf.abs(gradient_alt)
            )

            grad_sign = tf.sign(gradient)
            grad_alt_sign = tf.sign(gradient_alt)
            grad_same_sign = (grad_sign == grad_alt_sign) | (gradient == 0) | (gradient_alt == 0)
            grad_abs = tf.abs(gradient)
            grad_alt_abs = tf.abs(gradient_alt)

            gradient_scaled_mask = tf.cast(
                grad_same_sign,
                gradient.dtype
            )
            scaled_score = 1.0 + compared_gradients * self._alternate_distribution_tradeoff
            scaled_scores = gradient_scaled_mask * scaled_score

            gradient_negated_mask = tf.cast(
                tf.logical_not(grad_same_sign) & (grad_abs > grad_alt_abs),
                gradient.dtype
            )

            negated_score = compared_gradients * self._alternate_distribution_tradeoff
            negated_scores = gradient_negated_mask * negated_score

            scores = scaled_scores + negated_scores
            step_direction = tf.math.divide_no_nan(gradient, tf.norm(gradient))
            step = step_direction * norm * self._layer_scales[idx] * self._step_size
            scored_step = step * scores
            perturbation.assign(scored_step)


    def get_applied_layers_indices(self):
        return self._applied_layers



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
