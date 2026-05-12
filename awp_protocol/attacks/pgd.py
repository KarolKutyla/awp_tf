from typing import Any

import tensorflow as tf

from dataclasses import dataclass, replace

from awp_protocol.attacks.attack import TensorflowEvasionAttack


@dataclass(frozen=True)
class PGDParams:
    perturbation_bound: float = 8 / 255
    pgd_step: int = 10
    pgd_step_size: float = 2 / 255


class PGDAttack(TensorflowEvasionAttack):
    def __init__(
            self,
            model: tf.keras.Model,
            params: PGDParams | None = None,
            **overrides
    ):
        super().__init__(model)
        self._dtype = tf.float32
        self._params = params or PGDParams()
        self._params = replace(self._params, **overrides)

        self._perturbation_bound: tf.Tensor = tf.constant(self._params.perturbation_bound, dtype=self._dtype)
        self._pgd_step: int = self._params.pgd_step
        self._pgd_step_size: tf.Tensor = tf.constant(self._params.pgd_step_size, dtype=self._dtype)

    @tf.function
    def generate(self, x_batch: tf.Tensor, y_batch: tf.Tensor) -> tf.Tensor:
        x_adv = x_batch + tf.random.uniform(shape=x_batch.shape, minval=-self._perturbation_bound, maxval=self._perturbation_bound, dtype=self._dtype)
        for i in range(self._pgd_step):
            x_adv = self._pgd_iteration(x_batch, x_adv, y_batch)
            x_adv = tf.stop_gradient(x_adv)
        return x_adv

    def _pgd_iteration(self, x: tf.Tensor, x_adv: tf.Tensor, y: tf.Tensor) -> tf.Tensor:
        with tf.GradientTape() as tape:
            tape.watch(x_adv)
            logits = self.model(x_adv, training=True)
            loss = tf.keras.losses.sparse_categorical_crossentropy(y, logits, from_logits=True)
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
