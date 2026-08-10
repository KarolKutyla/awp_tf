from dataclasses import dataclass, replace

import tensorflow as tf
from tensorflow import keras

from awp_tf.attacks.attack import TensorflowEvasionAttack
from awp_tf.gentle_awp.weight_calculator import WeightCalculator, WeightParams
from awp_tf.losses import loss_context

from awp_tf.losses.loss import AdversarialLoss
from awp_tf.losses.loss_context import LossContext



@dataclass(frozen=True)
class AWPParams:
    alternate_iteration: int = 1
    awp_steps: int = 10
    weight_constraint: float = 5.0e-3
    step_size: float | None = None

    def calc_step_size(self):
        return self.weight_constraint / (self.awp_steps * self.alternate_iteration)



class BatchProcessor:

    def __init__(
            self,
            classifier: keras.Model,
            attack: TensorflowEvasionAttack,
            adversarial_loss: AdversarialLoss,
            tracked_layers: tuple[bool, ...],
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

        step_size = self._params.step_size or self._params.calc_step_size()
        weight_calculator_params = WeightParams(awp_steps=self._params.awp_steps, step_size=step_size)
        self._weight_calculator: WeightCalculator = WeightCalculator(self._classifier, tracked_layers, weight_calculator_params)
        self._alternate_iteration = tf.constant(self._params.alternate_iteration, dtype=tf.int32)


    @tf.function(jit_compile=True)
    def awp_train_step(self, x_batch, y_batch) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        self._weight_calculator.reset_weight_perturbations()
        loss_context = self._calc_non_training_loss_context(x_batch, y_batch, x_batch)
        last_calculated_adv = self._calc_weight_perturbation(loss_context)
        robust_loss, ctx = self._update_model_adversarial(x_batch, y_batch, last_calculated_adv)
        self._weight_calculator.subtract_weight_perturbations()

        clean_loss = self._clean_loss(y_true=y_batch, y_pred=ctx.logits_clean)
        return clean_loss, ctx.logits_clean, robust_loss, ctx.logits_adv


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


    def _calc_weight_perturbation(self, loss_context) -> tf.Tensor:
        i0 = tf.constant(0, dtype=tf.int32)
        invariant_shape = tf.TensorShape([None] + loss_context.x_batch.shape[1:])

        def cond(i, ctx: LossContext):
            return i < self._alternate_iteration

        def body(i, ctx: LossContext):
            x_adv = self._attack.generate(ctx.x_batch, ctx.y_batch)
            ctx = ctx._replace(x_adv=x_adv, logits_adv=self._classifier(x_adv, training=False))
            self._weight_calculator.calculate_weight_perturbation(ctx)
            return i + 1, ctx

        _, x_adv = tf.nest.map_structure(
            tf.stop_gradient,
            tf.while_loop(cond, body, [i0, loss_context], parallel_iterations=1, shape_invariants=[i0.get_shape(), invariant_shape])
        )
        return x_adv


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
