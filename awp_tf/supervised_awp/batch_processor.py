from dataclasses import dataclass, replace

import tensorflow as tf
from tensorflow import keras

from awp_tf.attacks.attack import EvasionAttack
from awp_tf.supervised_awp.weight_calculator import WeightCalculator, WeightParams

from awp_tf.losses.loss import AdversarialLoss



@dataclass(frozen=True)
class AWPParams:
    weight_constraint: float = 1.0e-2


class BatchProcessor:

    def __init__(
            self,
            classifier: keras.Model,
            attack: EvasionAttack,
            adversarial_loss: AdversarialLoss,
            tracked_layers: tuple[bool | float, ...],
            params: AWPParams | None = None,
            **overrides
    ):
        self._dtype : tf.dtypes.DType = classifier.weights[0].dtype
        self._params = params or AWPParams()
        self._params = replace(self._params, **overrides)

        self._classifier: tf.keras.Model = classifier
        _validate_optimizer(self._classifier)
        self._attack: EvasionAttack = attack
        self._robust_loss: AdversarialLoss = adversarial_loss
        self._clean_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

        weight_calculator_params = WeightParams(weight_constraint=self._params.weight_constraint)
        self._weight_calculator: WeightCalculator = WeightCalculator(self._classifier, self._robust_loss, tracked_layers, weight_calculator_params)

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
            robust_loss = self._robust_loss.calculate_weight_perturbation_loss(x_batch, y_batch, x_batch_adv, self._classifier, training=True)
        gradient = tape.gradient(robust_loss, self._classifier.trainable_variables)
        self._weight_calculator.restore_model()
        self._classifier.optimizer.apply(gradient)
        return clean_loss, logits_clean, adv_loss, logits_adv

    @tf.function(jit_compile=True)
    def adv_train_step(self, x_batch: tf.Tensor, y_batch:tf.Tensor) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        x_batch_adv = self._attack.generate(x_batch, y_batch)
        logits_clean = self._classifier(x_batch, training=False)
        clean_loss = self._clean_loss(y_true=y_batch, y_pred=logits_clean)
        logits_adv = self._classifier(x_batch_adv, training=False)
        adv_loss = self._clean_loss(y_true=y_batch, y_pred=logits_adv)

        robust_loss = self._update_model_adversarial(x_batch, y_batch, x_batch_adv)

        return clean_loss, logits_clean, adv_loss, logits_adv


    @tf.function(jit_compile=True)
    def validation_step(self, x_batch: tf.Tensor, y_batch: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        x_batch_adv = self._attack.generate(x_batch, y_batch)
        logits_clean = self._classifier(x_batch, training=False)
        clean_loss = self._clean_loss(y_true=y_batch, y_pred=logits_clean)
        logits_adv = self._classifier(x_batch_adv, training=False)
        adv_loss = self._clean_loss(y_true=y_batch, y_pred=logits_adv)

        return clean_loss, logits_clean, adv_loss, logits_adv


    def _update_model_adversarial(self, x, y, x_adv):
        with tf.GradientTape() as tape:
            robust_loss = self._robust_loss.calculate_weight_perturbation_loss(x, y, x_adv, self._classifier, training=True)
        gradient = tape.gradient(robust_loss, self._classifier.trainable_variables)
        self._classifier.optimizer.apply(gradient)
        return robust_loss


def _validate_optimizer(classifier: keras.models.Model):
    if classifier.optimizer is None:
        raise Exception("No optimizer provided for the classifier. For native awp compile your model with SGD with custom learning rate and 0.0 momentum.")

    if not classifier.optimizer.built:
        classifier.optimizer.build(classifier.trainable_variables)
