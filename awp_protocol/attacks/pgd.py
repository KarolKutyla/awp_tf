from typing import Any

import tensorflow as tf
from tensorflow import keras

from attacks.attack import TensorflowEvasionAttack


def get_default_params() -> dict:
    return {
        'perturbation_bound': 8 / 255,
        'pgd_step': 10,
        'pgd_step_size': 0.1
    }


class PGDAttack(TensorflowEvasionAttack):
    def __init__(
            self,
            model: keras.Model,
            params: dict[str, object] | Any = None
    ):
        super().__init__(model)
        self._forced_type = tf.float32

        params = _set_params(params)
        self._perturbation_bound: float = tf.cast(params['perturbation_bound'], self._forced_type)
        self._pgd_step: int = params['pgd_step']
        self._pgd_step_size: float = tf.cast(params['pgd_step_size'], self._forced_type)

    @tf.function
    def generate_attack(self, x_batch: tf.Tensor, y_batch: tf.Tensor) -> tf.Tensor:
        random_sample = tf.random.uniform(shape=x_batch.shape, minval=-1.0, maxval=1.0, dtype=self._forced_type)
        pert = random_sample * self._perturbation_bound
        x_adv = pert + x_batch
        for i in range(self._pgd_step):
            x_adv = self._pgd_iteration(x_batch, x_adv, y_batch)
        return x_adv

    @tf.function
    def _pgd_iteration(self, x: tf.Tensor, x_adv: tf.Tensor, y: tf.Tensor) -> tf.Tensor:
        with tf.GradientTape() as tape:
            tape.watch(x_adv)
            logits = self.model(x_adv, training=False)
            loss = tf.keras.losses.sparse_categorical_crossentropy(y, logits)
            loss = tf.reduce_mean(loss)
        gradient = tape.gradient(loss, x_adv)
        x_adv = x_adv + tf.sign(gradient) * self._pgd_step_size
        perturbation = x_adv - x
        perturbation = tf.clip_by_value(
            perturbation,
            -self._perturbation_bound,
            self._perturbation_bound
        )
        x_adv = x + perturbation
        x_adv = tf.clip_by_value(x_adv, 0.0, 1.0)
        return x_adv


def _set_params(training_params: dict[str, Any] | Any) -> dict[str, Any]:
    if training_params is None:
        return get_default_params()
    else:
        params_dict = get_default_params()
        params_dict.update(training_params)
        return params_dict
