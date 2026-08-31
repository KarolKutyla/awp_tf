import tensorflow as tf

from dataclasses import dataclass, replace

from awp_tf.attacks.attack import EvasionAttack
from awp_tf.losses.loss import AdversarialLoss


@dataclass(frozen=True)
class PGDParams:
    perturbation_bound: float = 8 / 255
    pgd_step: int = 10
    pgd_step_size: float = 2 / 255
    norm: str = "linf"


class PGDAttack(EvasionAttack):
    def __init__(
            self,
            model: tf.keras.Model,
            loss: AdversarialLoss,
            mean,
            std,
            params: PGDParams | None = None,
            **overrides
    ):
        super().__init__(model)
        self._loss = loss
        self._dtype = tf.float32
        self._params = params or PGDParams()
        self._params = replace(self._params, **overrides)
        self._perturbation_bound: tf.Tensor
        self._pgd_step_size: tf.Tensor

        self._pgd_step = tf.constant(self._params.pgd_step, dtype=tf.int32)
        self._perturbation_bound = tf.constant(self._params.perturbation_bound, dtype=self._dtype)
        self._pgd_step_size = tf.constant(self._params.pgd_step_size, dtype=self._dtype)
        self._norm = self._params.norm
        self._mean = tf.cast(mean, self._dtype)
        self._std = tf.cast(std, self._dtype)


    # @tf.function(reduce_retracing=True)
    def generate(self, x_batch: tf.Tensor, y_batch: tf.Tensor) -> tf.Tensor:
        if self._norm == "linf":
            return self._generate_inf(x_batch, y_batch)
        if self._norm == "l2":
            return self._generate_l2(x_batch, y_batch)
        if self._norm == "l1":
            raise Exception("Norm l1 not implemented")
        raise Exception(f"Unknown norm type: {self._params.norm}. Should be one of: linf, l2, l1.")


    def _generate_l2(self, x_batch: tf.Tensor, y_batch: tf.Tensor) -> tf.Tensor:
        norm_axes = tuple(range(1, len(x_batch.shape)))
        x = self._denormalize(x_batch)
        x_adv = self._random_sample_l2(x, norm_axes)

        def cond(i, x_adv):
            return i < self._pgd_step

        def body(i, x_adv):
            x_adv = self._pgd_l2_step(x, x_adv, y_batch, norm_axes)
            return i + 1, x_adv

        i0 = tf.constant(0, dtype=tf.int32)
        _, x_adv = tf.nest.map_structure(
            tf.stop_gradient,
            tf.while_loop(cond, body, [i0, x_adv], parallel_iterations=1))
        return self._normalize(x_adv)


    def _pgd_l2_step(self, x: tf.Tensor, x_adv: tf.Tensor, y: tf.Tensor, norm_indices: tuple) -> tf.Tensor:
        gradient = self._calculate_gradient(x, x_adv, y)
        gradient_norm = tf.sqrt(tf.reduce_sum(tf.square(gradient), axis=norm_indices, keepdims=True))
        gradient = (tf.math.divide_no_nan(gradient, gradient_norm))
        x_adv = x_adv + gradient * self._pgd_step_size

        perturbation = x_adv - x
        perturbation = self._project_l2(perturbation, norm_indices)
        x_adv = x + perturbation
        return tf.clip_by_value(x_adv, 0.0, 1.0)


    def _project_l2(self, perturbation, norm_indices: tuple):
        pert_norm = tf.sqrt(tf.reduce_sum(tf.square(perturbation), axis=norm_indices, keepdims=True))
        factor_ones = tf.ones_like(pert_norm)
        factor_bounds = tf.ones_like(pert_norm) * self._perturbation_bound

        factor = tf.minimum(
            factor_ones,
            tf.math.divide_no_nan(factor_bounds, pert_norm)
        )
        return perturbation * factor


    def _generate_inf(self, x_batch: tf.Tensor, y_batch: tf.Tensor) -> tf.Tensor:
        x = self._denormalize(x_batch)
        x_adv = self._random_sample_linf(x)

        def cond(i, x_adv):
            return i < self._pgd_step

        def body(i, x_adv):
            x_adv = self._pgd_linf_step(x, x_adv, y_batch)
            return i + 1, x_adv

        i0 = tf.constant(0, dtype=tf.int32)
        _, x_adv = tf.nest.map_structure(
            tf.stop_gradient,
            tf.while_loop(cond, body, [i0, x_adv], parallel_iterations=1))
        return self._normalize(x_adv)


    def _pgd_linf_step(self, x: tf.Tensor, x_adv: tf.Tensor, y: tf.Tensor) -> tf.Tensor:
        gradient = self._calculate_gradient(x, x_adv, y)
        gradient = tf.sign(gradient)
        x_adv = x_adv + gradient * self._pgd_step_size

        perturbation = x_adv - x
        perturbation = self._project_linf(perturbation)
        x_adv = x + perturbation
        return tf.clip_by_value(x_adv, 0.0, 1.0)


    def _project_linf(self, perturbation) -> tf.Tensor:
        return tf.clip_by_value(
            perturbation,
            -self._perturbation_bound,
            self._perturbation_bound
        )


    def _calculate_gradient(self, x: tf.Tensor, x_adv: tf.Tensor, y: tf.Tensor) -> tf.Tensor:
        model_scaled_x = self._normalize(x)
        with tf.GradientTape() as tape:
            tape.watch(x_adv)
            model_scaled_x_adv = self._normalize(x_adv)
            loss = self._loss.calculate_attack_loss(model_scaled_x, y, model_scaled_x_adv, self.model)
        gradient = tape.gradient(loss, x_adv)
        return gradient

    def _random_sample_l2(self, x_batch: tf.Tensor, norm_axes) -> tf.Tensor:
        shape = tf.shape(x_batch)
        delta = tf.random.normal(
            shape=shape,
            dtype=self._dtype
        )
        delta_norm = tf.sqrt(
            tf.reduce_sum(
                tf.square(delta),
                axis=norm_axes,
                keepdims=True
            )
        )
        delta = tf.math.divide_no_nan(delta, delta_norm)
        batch_size = tf.shape(x_batch)[0]
        r = tf.random.uniform(
            shape=(batch_size, 1, 1, 1),
            minval=0.0,
            maxval=1.0,
            dtype=self._dtype
        )
        delta = delta * r * self._perturbation_bound
        x_adv = x_batch + delta

        return tf.clip_by_value(x_adv, 0.0, 1.0)

    def _random_sample_linf(self, x_batch) -> tf.Tensor:
        x_adv = x_batch + tf.random.uniform(shape=tf.shape(x_batch), minval=-self._perturbation_bound, maxval=self._perturbation_bound, dtype=self._dtype)
        return tf.clip_by_value(x_adv, 0.0, 1.0)


    def project_l1(self):
        ...


    def _normalize(self, x: tf.Tensor):
        return (x - self._mean) / self._std


    def _denormalize(self, x: tf.Tensor):
        return x * self._std + self._mean
