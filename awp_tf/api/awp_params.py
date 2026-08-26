from dataclasses import dataclass, replace

import tensorflow as tf
from tensorflow import keras

@dataclass(frozen=True)
class AWPParams:

    steps: int = 1
    step_size: float = 1.e-2
    perturbation_size_constraint: float = 1.e-2

    alternate_iterations: int = 1

    alternate_distribution_tradeoff: float = 0.5


    def __post_init__(self):
        try:
            steps = int(self.steps)
        except (TypeError, ValueError):
            raise ValueError("steps must be convertible to int")
        if steps < 0:
            raise ValueError("steps must be non-negative")
        object.__setattr__(self, "steps", steps)

        if not 0 <= self.step_size:
            raise ValueError(
                "step_size must not be smaller than 0"
            )

        if not 0 <= self.perturbation_size_constraint:
            raise ValueError(
                "perturbation_size_constraint must not be smaller than 0"
            )

        try:
            alternate_iterations = int(self.alternate_iterations)
        except (TypeError, ValueError):
            raise ValueError("alternate_iterations must be convertible to int")
        if alternate_iterations < 0:
            raise ValueError("alternate_iterations must be non-negative")
        object.__setattr__(self, "alternate_iterations", alternate_iterations)

        if not 0 <= self.alternate_distribution_tradeoff <= 1:
            raise ValueError(
                "alternate_distribution_tradeoff must be between 0 and 1"
            )


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

