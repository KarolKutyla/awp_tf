from dataclasses import dataclass, replace

import tensorflow as tf
from tensorflow import keras
from tensorflow.python.training import training

from awp_protocol.attacks.attack import TensorflowEvasionAttack
from awp_protocol.attacks.pgd import PGDAttack
from awp_protocol.weight_calculator import WeightCalculator, WeightParams

from awp_protocol.losses.loss import AdversarialLoss
from awp_protocol.losses.loss_context import LossContext
from awp_protocol.losses import trades_loss, adversarial_categorical_cross_entropy



@dataclass(frozen=True)
class AWPParams:
    alternate_iteration: int = 1
    awp_steps: int = 10
    learning_rate: float = 0.01
    mode: str = "trades"
    use_optimizer: bool = True
    weight_constraint: float = 5.0e-3
    step_size: float = weight_constraint / (awp_steps * alternate_iteration)



class BatchProcessor:

    def __init__(
            self,
            classifier: keras.Model,
            proxy_classifier: keras.Model,
            tracked_layers: tuple[bool, ...],
            attack: TensorflowEvasionAttack | None = None,
            params: AWPParams | None = None,
            **overrides
    ):
        self._learning_rate: tf.Tensor


        self._params = params or AWPParams()
        self._params = replace(self._params, **overrides)

        self._classifier: tf.keras.Model = classifier
        self._proxy_classifier: keras.Model = proxy_classifier
        proxy_classifier_params = WeightParams(weight_constraint=self._params.weight_constraint, step_size=self._params.step_size)
        self._proxy_calculator: WeightCalculator = WeightCalculator(self._classifier, self._proxy_classifier, tracked_layers, proxy_classifier_params)
            #AWPProxyClassifier(self._classifier, tracked_layers, params_dict['weight_constraint']))

        self._clean_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        self._robust_loss = _select_adversarial_loss_from_params(self._params)

        self._trades_beta = 0.1
        self._dtype : tf.dtypes.DType = tf.float32

        self._attack_tf: TensorflowEvasionAttack = _select_attack(attack, proxy_classifier)

        self._learning_rate = tf.constant(self._params.learning_rate, dtype=self._dtype)
        self._use_optimizer: bool = self._params.use_optimizer
        self._alternate_iteration = self._params.alternate_iteration
        self._awp_steps = self._params.awp_steps


    @tf.function(jit_compile=True)
    def awp_train_step(self, x_batch, y_batch) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        self._proxy_calculator.batch_process_begin()
        x_adv = x_batch
        for a in range(self._alternate_iteration):
            x_adv = self._attack_tf.generate(x_batch, y_batch)
            self._find_weight_perturbation(x_batch, y_batch, x_adv)

        self._proxy_calculator.add_weight_perturbations()
        with tf.GradientTape() as tape:
            ctx = self._training_forward_pass(x_batch, y_batch, x_adv)
            robust_loss = self._robust_loss.calculate(ctx)
        gradient = tape.gradient(robust_loss, self._classifier.trainable_variables)
        self._update_classifier(gradient)
        self._proxy_calculator.subtract_weight_perturbations()

        clean_loss = self._clean_loss(y_true=y_batch, y_pred=ctx.logits_clean)
        return clean_loss, ctx.logits_clean, robust_loss, ctx.logits_adv


    @tf.function(jit_compile=True)
    def adv_train_step(self, x_batch, y_batch) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        x_adv = self._attack_tf.generate(x_batch, y_batch)
        with tf.GradientTape() as tape:
            ctx = self._training_forward_pass(x_batch, y_batch, x_adv)
            robust_loss = self._robust_loss.calculate(ctx)
        gradient = tape.gradient(robust_loss, self._classifier.trainable_variables)
        self._update_classifier(gradient)

        clean_loss = self._clean_loss(y_true=y_batch, y_pred=ctx.logits_clean)
        return clean_loss, ctx.logits_clean, robust_loss, ctx.logits_adv


    @tf.function(jit_compile=True)
    def validation_step(self, x_batch: tf.Tensor, y_batch: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        x_adv = self._attack_tf.generate(x_batch, y_batch)
        ctx = self._validation_forward_pass(x_batch, y_batch, x_adv)
        clean_loss = self._clean_loss(y_batch, ctx.logits_clean)
        robust_loss = self._robust_loss.calculate(ctx)
        return clean_loss, ctx.logits_clean, robust_loss, ctx.logits_adv


    def _find_weight_perturbation(self, x_batch: tf.Tensor, y_batch: tf.Tensor, x_pert: tf.Tensor):
        for j in range(self._awp_steps):
            self._weight_perturbation_step(x_batch, y_batch, x_pert)


    def _weight_perturbation_step(self, x_batch: tf.Tensor, y_batch: tf.Tensor, x_pert: tf.Tensor):
        with tf.GradientTape() as tape:
            result = self._proxy_forward_pass(x_batch, y_batch, x_pert)
            loss = self._robust_loss.calculate(result)
        gradient = tape.gradient(loss, self._proxy_calculator.trainable_variables)
        self._proxy_calculator.calculate_and_update_weight_perturbation(gradient)


    def _training_forward_pass(self, x_batch: tf.Tensor, y_batch: tf.Tensor, x_pert: tf.Tensor) -> LossContext:
        logits = self._classifier(x_batch, training=True)
        logits_adv = self._classifier(x_pert, training=True)
        ctx = LossContext(
            x_batch=x_batch,
            x_adv=x_pert,
            y_batch=y_batch,
            logits_clean=logits,
            logits_adv=logits_adv
        )
        return ctx


    def _validation_forward_pass(self, x_batch: tf.Tensor, y_batch: tf.Tensor, x_pert: tf.Tensor) -> LossContext:
        logits = self._classifier(x_batch, training=False)
        logits_adv = self._classifier(x_pert, training=False)
        ctx = LossContext(
            x_batch=x_batch,
            x_adv=x_pert,
            y_batch=y_batch,
            logits_clean=logits,
            logits_adv=logits_adv
        )
        return ctx


    def _proxy_forward_pass(self, x_batch: tf.Tensor, y_batch: tf.Tensor, x_pert: tf.Tensor) -> LossContext:
        logits = self._proxy_classifier(x_batch, training=True)
        logits_adv = self._proxy_classifier(x_pert, training=True)
        ctx = LossContext(
            x_batch=x_batch,
            x_adv=x_pert,
            y_batch=y_batch,
            logits_clean=logits,
            logits_adv=logits_adv
        )
        return ctx


    def _update_classifier(self, gradients: list[tf.Tensor]):
        variables = self._classifier.trainable_variables
        if self._use_optimizer and self._classifier.optimizer is not None:
            self._classifier.optimizer.apply(gradients)
        else:
            for gradient, variable in zip(gradients, variables):
                if gradient is None:
                    gradient = tf.zeros_like(variable)
                variable.assign_sub(self._learning_rate * gradient)



def _create_storage_for_calculated_perturbations(classifier: keras.models.Model):
    return [tf.Variable(tf.zeros_like(variable), trainable=False) for variable in classifier.trainable_weights]


def _select_awp_step_size(params_dict: dict[str, object]):
    if params_dict["awp_step_size"] is None:
        return params_dict['weight_constraint'] / (params_dict['awp_steps'] * params_dict['alternate_iteration'])
    else:
        return params_dict["awp_step_size"]


def _clone_init_optimizer_for_proxy_classifier(proxy_classifier, optimizer, learning_rate):
    cfg = optimizer.get_config()
    opt = optimizer.__class__.from_config(cfg)
    zero_grads = [tf.zeros_like(v) for v in proxy_classifier.trainable_variables]
    opt.learning_rate.assign(learning_rate)
    opt.apply_gradients(zip(zero_grads, proxy_classifier.trainable_variables))
    return opt


def _select_attack(attack: TensorflowEvasionAttack | None, proxy_classifier: keras.Model) -> TensorflowEvasionAttack:
    if attack is None:
        return PGDAttack(proxy_classifier)
    if isinstance(attack, TensorflowEvasionAttack):
        return attack
    else:
        raise Exception(f"Invalid type of attack: {type(attack)}")


def _select_adversarial_loss_from_params(params: AWPParams) -> AdversarialLoss:
    if params.mode == "pgd":
        return adversarial_categorical_cross_entropy.AdversarialSparseCategoricalCrossEntropy()
    elif params.mode == "trades":
        return trades_loss.TradesLoss()
    else:
        raise Exception("Mode not provided! Chose pgd or trades.")
