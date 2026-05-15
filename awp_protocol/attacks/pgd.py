from typing import Any

import tensorflow as tf

from dataclasses import dataclass, replace

from awp_protocol.attacks.attack import TensorflowEvasionAttack


@dataclass(frozen=True)
class PGDParams:
    perturbation_bound: float = 8 / 255
    pgd_step: int = 10
    pgd_step_size: float = 2 / 255
    norm: str = "linf"


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

    @tf.function(reduce_retracing=True)
    def generate(self, x_batch: tf.Tensor, y_batch: tf.Tensor) -> tf.Tensor:
        if self._params.norm == "linf":
            return self._generate_inf(x_batch, y_batch)
        if self._params.norm == "l2":
            return self._generate_l2(x_batch, y_batch)
        raise Exception(f"Unknown norm type: {self._params.norm}. Should be one of: linf, l2, l1.")


    def _generate_l2(self, x_batch: tf.Tensor, y_batch: tf.Tensor) -> tf.Tensor:
        x_adv = self._random_sample(x_batch)
        for i in range(self._pgd_step):
            x_adv = self._pgd_l2_iteration(x_batch, x_adv, y_batch)
        return x_adv


    def _pgd_l2_iteration(self, x: tf.Tensor, x_adv: tf.Tensor, y: tf.Tensor) -> tf.Tensor:
        gradient = self._gradient(x_adv, y)
        flat = tf.reshape(gradient, [tf.shape(gradient)[0], -1])
        grad_norm = tf.norm(flat, axis=1, keepdims=True)
        gradient = (tf.math.divide_no_nan(gradient, grad_norm))
        x_adv = x_adv + gradient * self._pgd_step_size

        perturbation = x_adv - x
        perturbation = self._project_l2(perturbation)
        x_adv = x + perturbation
        x_adv = tf.clip_by_value(x_adv, 0.0, 1.0)
        return x_adv


    def _project_l2(self, perturbation):
        flat = tf.reshape(perturbation, [tf.shape(perturbation)[0], -1])

        pert_norm = tf.norm(flat, axis=1, keepdims=True)

        factor = tf.minimum(
            1.0,
            self._perturbation_bound / (pert_norm + 1e-12)
        )
        return perturbation * factor


    def _generate_inf(self, x_batch: tf.Tensor, y_batch: tf.Tensor) -> tf.Tensor:
        x_adv = self._random_sample(x_batch)
        for i in range(self._pgd_step):
            x_adv = self._pgd_linf_iteration(x_batch, x_adv, y_batch)
        return x_adv


    def _pgd_linf_iteration(self, x: tf.Tensor, x_adv: tf.Tensor, y: tf.Tensor) -> tf.Tensor:
        gradient = self._gradient(x_adv, y)
        gradient = tf.sign(gradient)
        x_adv = x_adv + gradient * self._pgd_step_size

        perturbation = x_adv - x
        perturbation = self._project_linf(perturbation)
        x_adv = x + perturbation
        x_adv = tf.clip_by_value(x_adv, 0.0, 1.0)
        return x_adv


    def _project_linf(self, perturbation) -> tf.Tensor:
        return tf.clip_by_value(
            perturbation,
            -self._perturbation_bound,
            self._perturbation_bound
        )


    def _gradient(self, x_adv: tf.Tensor, y: tf.Tensor) -> tf.Tensor:
        with tf.GradientTape() as tape:
            tape.watch(x_adv)
            logits = self.model(x_adv, training=False)
            loss = tf.keras.losses.sparse_categorical_crossentropy(y, logits, from_logits=True)
            loss = tf.reduce_mean(loss)
        gradient = tape.gradient(loss, x_adv)
        tf.stop_gradient(gradient)
        return gradient


    def _random_sample(self, x_batch) -> tf.Tensor:
        x_adv = x_batch + tf.random.uniform(shape=tf.shape(x_batch), minval=-self._perturbation_bound, maxval=self._perturbation_bound, dtype=self._dtype)
        return tf.clip_by_value(x_adv, 0.0, 1.0)


    def project_l1(self):
        ...
