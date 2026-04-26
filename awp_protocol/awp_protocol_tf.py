from typing import Optional

import tensorflow as tf
from tensorflow import keras
from tensorflow.python.eager.def_function import Function as TfFunction
import numpy as np


from tensorflow.python.types.core import GenericFunction

from awp_protocol.attacks import TensorflowEvasionAttack
from awp_protocol.attacks.pgd import PGDAttack
from awp_protocol.losses.trades_loss import TradesLoss
from awp_proxy import AWPProxyClassifier

from neural_network_analytic_tool.art_tf.losses.loss import AdversarialAttackLoss
from neural_network_analytic_tool.art_tf.losses.loss_context import LossContext
from neural_network_analytic_tool.art_tf.losses import trades_loss, adversarial_categorical_cross_entropy


def get_default_params() -> dict:
    return {
        'weight_constraint': 0.01,
        'alternate_iteration': 1,
        'learning_rate': 0.01,
        'awp_steps': 10,
        "awp_step_size": 0.1,
        "mode": "trades"
    }


class AWPProtocolTF:
    EPS = 1e-8

    def __init__(
            self,
            classifier: keras.Model,
            attack: Optional[TensorflowEvasionAttack] = None,
            optimizer: Optional[keras.optimizers.Optimizer] = None,
            tracked_layers: list[bool] | None = None,
            params_dict: Optional[dict] = None
    ):
        params_dict = _set_params(params_dict)
        self._classifier = classifier
        self._loss = _select_adversarial_loss_from_params(params_dict)
        self._trades_beta = 0.1

        self._attack: TensorflowEvasionAttack = _set_attack(attack, classifier)

        self._learning_rate = params_dict["learning_rate"]
        self._optimizer: tf.optimizers.Optimizer = optimizer
        self._weight_constraint = params_dict["weight_constraint"]
        self._alternate_iteration = params_dict["alternate_iteration"]
        self._awp_steps = params_dict["awp_steps"]
        self._awp_step_size = _select_awp_step_size(params_dict)

        self._perturbed_layers = _select_perturbed_layers(self._classifier, tracked_layers)
        self._weight_perturbations = _create_storage_for_calculated_perturbations(self._classifier)
        self._weight_norms = [tf.Variable(tf.norm(variables)) if tracked else None for variables, tracked in
                              zip(self._classifier.trainable_variables, self._perturbed_layers)]
        self._weight_perturbation_sizes = [tf.Variable(weight_size * self._weight_constraint) if tracked else None
                                           for weight_size, tracked in
                                           zip(self._weight_norms, self._perturbed_layers)]
        self._proxy_classifier: AWPProxyClassifier = \
            AWPProxyClassifier(self._classifier, tracked_layers, params_dict['weight_constraint'])

    def batch_process_metrics(self, x_batch: tf.Tensor, y_batch: tf.Tensor):
        result = self.batch_process(x_batch, y_batch)
        loss_adv = result["loss"]
        logits_adv = result["logits_adv"]

        y_pred = tf.argmax(logits_adv, axis=1)
        y_true = tf.argmax(y_batch, axis=1)
        correct_predictions = tf.reduce_sum(tf.cast(y_pred == y_true, dtype=tf.float32))

        batch_size = tf.cast(x_batch.shape[0], dtype=tf.float32)
        accuracy = correct_predictions / batch_size
        return loss_adv, accuracy, batch_size

    @tf.function
    def batch_process(self, x_batch: tf.Tensor, y_batch: tf.Tensor) -> dict:
        self._proxy_classifier.copy_originator_state(self._classifier)
        x_pert = x_batch
        for a in range(self._alternate_iteration):
            x_pert = self._attack.generate_attack(x_batch, y_batch)
            self._find_weight_perturbation(x_batch, y_batch, x_pert)

        with tf.GradientTape() as tape:
            ctx = self._feed_proxy(x_batch, y_batch, x_pert)
            loss = self._loss.calculate(ctx)

        gradient = tape.gradient(loss, self._proxy_classifier.get_trainable_variables())
        self._update_classifier(gradient)
        return ctx

    @tf.function
    def _find_weight_perturbation(self, x_batch: tf.Tensor, y_batch: tf.Tensor, x_pert: tf.Tensor):
        for j in range(self._awp_steps):
            self._weight_perturbation_step(x_batch, y_batch, x_pert)

    @tf.function
    def _weight_perturbation_step(self, x_batch, y_batch, x_pert):
        with tf.GradientTape() as tape:
            result = self._feed_proxy(x_batch, y_batch, x_pert)
            loss = self._loss.calculate(result)
        gradient = tape.gradient(loss, self._proxy_classifier.get_trainable_variables())
        self._proxy_classifier.calculate_and_apply_weight_perturbations(gradient, self._classifier)

    @tf.function
    def _feed_proxy(self, x_batch: tf.Tensor, y_batch: tf.Tensor, x_pert: tf.Tensor) -> LossContext:
        logits = self._proxy_classifier.forward_pass(x_batch)
        logits_adv = self._proxy_classifier.forward_pass(x_pert)
        ctx = LossContext(x_batch, x_pert, logits, logits_adv, y_batch)
        return ctx

    @tf.function
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


def _set_params(training_params: Optional[dict]) -> dict:
    if training_params is None:
        return get_default_params()
    else:
        params_dict = get_default_params()
        params_dict.update(training_params)
        return params_dict


def _select_perturbed_layers(classifier, tracked_layers) -> list[bool]:
    if tracked_layers is None:
        return ['kernel' in variable.name for variable in classifier.trainable_variables]
    else:
        return tracked_layers


def _create_storage_for_calculated_perturbations(classifier: keras.models.Model):
    return [tf.Variable(tf.zeros_like(variable), trainable=False) for variable in classifier.trainable_weights]


def _select_awp_step_size(params_dict: dict[str: object]):
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


def _set_attack(attack: Optional[TensorflowEvasionAttack], model: keras.Model) -> TensorflowEvasionAttack:
    if attack is None:
        adv_loss = TradesLoss(model)
        return PGDAttack(model, adv_loss)
    else:
        return attack


def _select_adversarial_loss_from_params(params_dict) -> AdversarialAttackLoss:
    if params_dict["mode"] == "pgd":
        return adversarial_categorical_cross_entropy.AdversarialCategoricalCrossEntropy()
    elif params_dict["mode"] == "trades":
        return trades_loss.TradesLoss()
    else:
        raise Exception("Mode not provided! Chose pgd or trades.")
