from dataclasses import dataclass, replace

import tensorflow as tf
from tensorflow import keras

from awp_tf.losses.loss import AdversarialLoss
from awp_tf.attacks.attack import EvasionAttack
from awp_tf.api.awp_operations import Calculator


class AWP:
    def __init__(self, calculator: Calculator, attack: EvasionAttack):
        self._calculator = calculator
        self._attack = attack

    @tf.function(jit_compile=True)
    def awp_train_step(self, x_batch: tf.Tensor, y_batch: tf.Tensor, x_batch_alt: tf.Tensor, y_batch_alt: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        x_batch_adv = self._attack.generate(x_batch, y_batch)
        x_batch_adv_alt = self._attack.generate(x_batch_alt, y_batch_alt)
        logits_clean = self._classifier(x_batch, training=False)
        clean_loss = self._clean_loss(y_true=y_batch, y_pred=logits_clean)
        logits_adv = self._classifier(x_batch_adv, training=False)
        adv_loss = self._clean_loss(y_true=y_batch, y_pred=logits_adv)

        self._weight_calculator.initiate_state_for_batch_process()
        self._weight_calculator.calculate_weight_perturbation(x_batch, y_batch, x_batch_adv, x_batch_alt, y_batch_alt, x_batch_adv_alt)
        self._weight_calculator.apply_weight_perturbations()

        with tf.GradientTape() as tape:
            robust_loss = self._robust_loss.calculate(x_batch, y_batch, x_batch_adv, self._classifier, training=True)
        gradient = tape.gradient(robust_loss, self._classifier.trainable_variables)
        self._weight_calculator.restore_model()
        self._classifier.optimizer.apply(gradient)
        return clean_loss, logits_clean, adv_loss, logits_adv
