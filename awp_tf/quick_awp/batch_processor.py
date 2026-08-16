from dataclasses import dataclass, replace

import tensorflow as tf
from tensorflow import keras

from awp_tf.attacks.attack import TensorflowEvasionAttack
from awp_tf.quick_awp.weight_calculator import WeightCalculator, WeightParams

from awp_tf.losses.loss import AdversarialLoss
from awp_tf.losses.loss_context import LossContext



@dataclass(frozen=True)
class AWPParams:
    weight_constraint: float = 1.0e-2


class BatchProcessor:

    def __init__(
            self,
            classifier: keras.Model,
            attack: TensorflowEvasionAttack,
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
        self._attack: TensorflowEvasionAttack = attack
        self._robust_loss: AdversarialLoss = adversarial_loss
        self._clean_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

        weight_calculator_params = WeightParams(weight_constraint=self._params.weight_constraint)
        self._weight_calculator: WeightCalculator = WeightCalculator(self._classifier, tracked_layers, weight_calculator_params, loss=self._robust_loss)


    @tf.function(jit_compile=True)
    def awp_train_step(self, x_batch: tf.Tensor, y_batch: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:

        x_batch_adv = self._attack.generate(x_batch, y_batch)
        logits_clean = self._classifier(x_batch, training=False)
        clean_loss = self._clean_loss(y_true=y_batch, y_pred=logits_clean)
        logits_adv = self._classifier(x_batch_adv, training=False)
        adv_loss = self._clean_loss(y_true=y_batch, y_pred=logits_adv)

        self._weight_calculator.initiate_state_for_batch_process()
        self._weight_calculator.calculate_weight_perturbation(x_batch, y_batch, x_batch_adv)
        self._weight_calculator.apply_weight_perturbations()

        with tf.GradientTape() as tape:
            ctx = self._calc_training_loss_context(x_batch, y_batch, x_batch_adv)
            robust_loss = self._robust_loss.calculate(ctx)
        gradient = tape.gradient(robust_loss, self._classifier.trainable_variables)
        self._weight_calculator.restore_model()
        self._classifier.optimizer.apply(gradient)
        return clean_loss, logits_clean, adv_loss, logits_adv


    @tf.function(jit_compile=True)
    def adv_train_step(self, x_batch, y_batch) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        x_adv = self._attack.generate(x_batch, y_batch)
        robust_loss, ctx = self._update_model_adversarial(x_batch, y_batch, x_adv)

        clean_loss = self._clean_loss(y_true=y_batch, y_pred=ctx.logits_clean)
        return clean_loss, ctx.logits_clean, robust_loss, ctx.logits_adv


    @tf.function(jit_compile=True)
    def validation_step(self, x_batch: tf.Tensor, y_batch: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        x_adv = self._attack.generate(x_batch, y_batch)
        ctx = self._calc_non_training_loss_context(x_batch, y_batch, x_adv)
        clean_loss = self._clean_loss(y_batch, ctx.logits_clean)
        robust_loss = self._robust_loss.calculate(ctx)
        return clean_loss, ctx.logits_clean, robust_loss, ctx.logits_adv


    def _update_model_adversarial(self, x, y, x_adv):
        with tf.GradientTape() as tape:
            ctx = self._calc_training_loss_context(x, y, x_adv)
            robust_loss = self._robust_loss.calculate(ctx)
        gradient = tape.gradient(robust_loss, self._classifier.trainable_variables)
        self._classifier.optimizer.apply(gradient)
        return robust_loss, ctx



    def _calc_training_loss_context(self, x: tf.Tensor, y: tf.Tensor, x_adv: tf.Tensor) -> LossContext:
        return self._calc_loss_context(x, y, x_adv, True)


    def _calc_non_training_loss_context(self, x: tf.Tensor, y: tf.Tensor, x_adv: tf.Tensor) -> LossContext:
        return self._calc_loss_context(x, y, x_adv, False)


    def _calc_loss_context(self, x: tf.Tensor, y: tf.Tensor, x_adv: tf.Tensor, training: bool):
        batch_size = tf.shape(x)[0]
        xx = tf.concat([x, x_adv], axis=0)
        logits = self._classifier(xx, training=training)
        logits_clean = logits[:batch_size]
        logits_adv = logits[batch_size:]
        ctx = LossContext(
            x_batch=x,
            x_adv=x_adv,
            y_batch=y,
            logits_clean=logits_clean,
            logits_adv=logits_adv
        )
        return ctx


def _validate_optimizer(classifier: keras.models.Model):
    if classifier.optimizer is None:
        raise Exception("No optimizer provided for the classifier. For native awp compile your model with SGD with custom learning rate and 0.0 momentum.")

    if not classifier.optimizer.built:
        classifier.optimizer.build(classifier.trainable_variables)
