from dataclasses import dataclass, replace

import tensorflow as tf
from tensorflow import keras

from awp_protocol.attacks.attack import TensorflowEvasionAttack
from awp_protocol.attacks.pgd import PGDAttack
from awp_protocol.losses.trades_loss import TradesLoss
from awp_protocol.awp_proxy import AWPProxyCalculations, AWPProxyParams

from awp_protocol.losses.loss import AdversarialLoss
from awp_protocol.losses.loss_context import LossContext
from awp_protocol.losses import trades_loss, adversarial_categorical_cross_entropy

@dataclass(frozen=True)
class AWPProtocolParams:
    alternate_iteration: int = 1
    learning_rate: float = 0.01
    awp_steps: int = 10
    mode: str = "trades"
    use_optimizer: bool = False
    proxy_params: AWPProxyParams = AWPProxyParams()


class AWPProtocolTF:

    def __init__(
            self,
            classifier: keras.Model,
            proxy_classifier: keras.Model,
            tracked_layers: tuple[bool, ...],
            attack: TensorflowEvasionAttack | None = None,
            optimizer: keras.optimizers.Optimizer | None = None,
            params: AWPProtocolParams | None = None,
            **overrides
    ):
        self._params = params or AWPProtocolParams()
        self._params = replace(self._params, **overrides)
        self._classifier: tf.keras.Model = classifier
        self._proxy_classifier: keras.Model = proxy_classifier
        self._proxy_calculator: AWPProxyCalculations = AWPProxyCalculations(self._classifier, self._proxy_classifier, tracked_layers, self._params.proxy_params)
            #AWPProxyClassifier(self._classifier, tracked_layers, params_dict['weight_constraint']))

        self._adversarial_loss = _select_adversarial_loss_from_params(self._params)
        self._trades_beta = 0.1
        self._dtype : tf.dtypes.DType = tf.float32

        self._attack_tf: TensorflowEvasionAttack = _select_attack(attack, proxy_classifier)

        self._learning_rate: tf.Tensor = tf.cast(self._params.learning_rate, dtype=self._dtype)
        self._optimizer: tf.optimizers.Optimizer | None = optimizer if optimizer is not None and self._params.use_optimizer else None

        self._alternate_iteration = self._params.alternate_iteration
        self._awp_steps = self._params.awp_steps

    # @tf.function
    def batch_process(self, x_batch: tf.Tensor, y_batch: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        self._proxy_calculator.batch_process_begin()
        x_pert = x_batch
        for a in range(self._alternate_iteration):
            x_pert = self._attack_tf.generate(x_batch, y_batch)
            self._find_weight_perturbation(x_batch, y_batch, x_pert)

        with tf.GradientTape() as tape:
            ctx = self._proxy_forward_pass(x_batch, y_batch, x_pert)
            loss = self._adversarial_loss.calculate(ctx)
        gradient = tape.gradient(loss, self._proxy_calculator.trainable_variables)
        self._update_classifier(gradient)
        return loss, ctx.logits_pert


    # @tf.function
    def _find_weight_perturbation(self, x_batch: tf.Tensor, y_batch: tf.Tensor, x_pert: tf.Tensor):
        for j in range(self._awp_steps):
            self._weight_perturbation_step(x_batch, y_batch, x_pert)

    # @tf.function
    def _weight_perturbation_step(self, x_batch: tf.Tensor, y_batch: tf.Tensor, x_pert: tf.Tensor):
        with tf.GradientTape() as tape:
            result = self._proxy_forward_pass(x_batch, y_batch, x_pert)
            loss = self._adversarial_loss.calculate(result)
        gradient = tape.gradient(loss, self._proxy_calculator.trainable_variables)
        self._proxy_calculator.calculate_and_update_weight_perturbation(gradient)

    # @tf.function
    def _proxy_forward_pass(self, x_batch: tf.Tensor, y_batch: tf.Tensor, x_pert: tf.Tensor) -> LossContext:
        logits = self._proxy_calculator.forward_pass(x_batch)
        logits_adv = self._proxy_calculator.forward_pass(x_pert)
        ctx = LossContext(
            x_batch=x_batch,
            x_pert=x_pert,
            y_true=y_batch,
            logits_out=logits,
            logits_pert=logits_adv
        )
        return ctx

    # @tf.function
    def _update_classifier(self, gradients: list[tf.Tensor]):
        variables = self._classifier.trainable_variables
        if self._optimizer is not None:
            grads_and_vars = [
                (g if g is not None else tf.zeros_like(v), v)
                for g, v in zip(gradients, variables)
            ]
            self._optimizer.apply_gradients(grads_and_vars)
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
        adv_loss = TradesLoss()
        return PGDAttack(proxy_classifier, adv_loss)
    if isinstance(attack, TensorflowEvasionAttack):
        return attack
    else:
        raise Exception(f"Invalid type of attack: {type(attack)}")


def _select_adversarial_loss_from_params(params: AWPProtocolParams) -> AdversarialLoss:
    if params.mode == "pgd":
        return adversarial_categorical_cross_entropy.AdversarialSparseCategoricalCrossEntropy()
    elif params.mode == "trades":
        return trades_loss.TradesLoss()
    else:
        raise Exception("Mode not provided! Chose pgd or trades.")
