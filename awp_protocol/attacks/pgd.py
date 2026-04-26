from typing import Optional

import tensorflow as tf
from tensorflow import keras

from neural_network_analytic_tool.art_tf.losses.loss import AdversarialAttackLoss
from attack import TensorflowEvasionAttack


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
            adversarial_loss: AdversarialAttackLoss,
            params: Optional[dict[str: object]] = None
    ):
        super().__init__()
        self._model: keras.Model = model
        self._adversarial_loss: AdversarialAttackLoss = adversarial_loss

        params = _set_params(params)
        self._perturbation_bound = params['perturbation_bound']
        self._pgd_step = params['pgd_step']
        self._pgd_step_size = params['pgd_step_size']

    @tf.function
    def generate_attack(self, x_batch: tf.Tensor, y_batch: tf.Tensor) -> tf.Tensor:
        random_sample = tf.random.uniform(shape=x_batch.shape, minval=-1.0, maxval=1.0, dtype=tf.float32)
        pert = random_sample * self._perturbation_bound
        x_adv = pert + x_batch
        for i in range(self._pgd_step):
            x_adv = self._pgd_iteration(x_batch, x_adv, y_batch)
        return x_adv

    @tf.function
    def _pgd_iteration(self, x, x_adv, y):
        with tf.GradientTape() as tape:
            tape.watch(x_adv)
            adversarial_loss = self._adversarial_loss.calculate(x, y, x_adv)
        gradient = tape.gradient(adversarial_loss, x_adv)
        pert = (x_adv - x) + gradient * self._pgd_step_size
        pert = tf.map_fn(fn=self._trim_pert_to_bound, elems=pert)
        return x + pert

    @tf.function
    def _trim_pert_to_bound(self, t):
        norm = tf.norm(t)
        return tf.cond(norm > self._perturbation_bound, lambda: t * self._perturbation_bound / norm, lambda: t)


def _set_params(training_params: Optional[dict[str: object]]) -> dict[str: object]:
    if training_params is None:
        return get_default_params()
    else:
        params_dict = get_default_params()
        params_dict.update(training_params)
        return params_dict
