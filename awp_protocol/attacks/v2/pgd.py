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

        self._pgd_step = tf.constant(self._params.pgd_step, dtype=tf.int32)
        self._perturbation_bound: tf.Tensor = tf.constant(self._params.perturbation_bound * 2.0, dtype=self._dtype)
        self._pgd_step_size: tf.Tensor = tf.constant(self._params.pgd_step_size * 2.0, dtype=self._dtype)


    @tf.function(reduce_retracing=True)
    def generate(self, x_batch: tf.Tensor, y_batch: tf.Tensor) -> tf.Tensor:
        if self._params.norm == "linf":
            return self._generate_inf(x_batch, y_batch)
        if self._params.norm == "l2":
            return self._generate_l2(x_batch, y_batch)
        if self._params.norm == "l1":
            raise Exception("Norm l1 not implemented")
        raise Exception(f"Unknown norm type: {self._params.norm}. Should be one of: linf, l2, l1.")


    def _generate_l2(self, x_batch: tf.Tensor, y_batch: tf.Tensor) -> tf.Tensor:
        x_adv = self._random_sample(x_batch)
        i0 = tf.constant(0, dtype=tf.int32)
        invariant_shape = tf.TensorShape([None] + x_batch.shape[1:])
        norm_indices = tuple(range(1, len(x_batch.shape)))
        def cond(i, x):
            return i < self._pgd_step

        def body(i, x):
            x = self._pgd_l2_iteration(x_batch, x, y_batch, norm_indices)
            return i + 1, x

        _, x_adv = tf.nest.map_structure(
            tf.stop_gradient,
            tf.while_loop(cond, body, [i0, x_adv], parallel_iterations=1, shape_invariants=[i0.get_shape(), invariant_shape]))
        return x_adv


    def _pgd_l2_iteration(self, x: tf.Tensor, x_adv: tf.Tensor, y: tf.Tensor, norm_indices: tuple) -> tf.Tensor:
        gradient = self._calculate_gradient(x_adv, y)
        gradient_norm = tf.sqrt(tf.reduce_sum(tf.square(gradient), axis=norm_indices, keepdims=True))
        gradient = (tf.math.divide_no_nan(gradient, gradient_norm))
        x_adv = x_adv + gradient * self._pgd_step_size

        perturbation = x_adv - x
        perturbation = self._project_l2(perturbation, norm_indices)
        x_adv = x + perturbation
        x_adv = tf.clip_by_value(x_adv, -1.0, 1.0)
        return x_adv


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
        x_adv = self._random_sample(x_batch)
        for i in range(self._pgd_step):
            x_adv = self._pgd_linf_iteration(x_batch, x_adv, y_batch)
        return x_adv


    def _pgd_linf_iteration(self, x: tf.Tensor, x_adv: tf.Tensor, y: tf.Tensor) -> tf.Tensor:
        gradient = self._calculate_gradient(x_adv, y)
        gradient = tf.sign(gradient)
        x_adv = x_adv + gradient * self._pgd_step_size

        perturbation = x_adv - x
        perturbation = self._project_linf(perturbation)
        x_adv = x + perturbation
        x_adv = tf.clip_by_value(x_adv, -1.0, 1.0)
        return x_adv


    def _project_linf(self, perturbation) -> tf.Tensor:
        return tf.clip_by_value(
            perturbation,
            -self._perturbation_bound,
            self._perturbation_bound
        )


    def _calculate_gradient(self, x_adv: tf.Tensor, y: tf.Tensor) -> tf.Tensor:
        with tf.GradientTape() as tape:
            tape.watch(x_adv)
            logits = self.model(x_adv, training=False)
            loss = tf.keras.losses.sparse_categorical_crossentropy(y, logits, from_logits=True)
            loss = tf.reduce_mean(loss)
        gradient = tape.gradient(loss, x_adv)
        return gradient


    def _random_sample(self, x_batch) -> tf.Tensor:
        x_adv = x_batch + tf.random.uniform(shape=tf.shape(x_batch), minval=-self._perturbation_bound, maxval=self._perturbation_bound, dtype=self._dtype)
        return tf.clip_by_value(x_adv, -1.0, 1.0)


    def project_l1(self):
        ...
