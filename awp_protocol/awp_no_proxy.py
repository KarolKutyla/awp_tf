# MIT License
#
# Copyright (C) The Adversarial Robustness Toolbox (ART) Authors 2023
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit
# persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
# Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
# WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""
This is a TensorFlow implementation of the Adversarial Weight Perturbation (AWP) protocol.

| Paper link: https://proceedings.neurips.cc/paper/2020/file/1ef91c212e30e14bf125e9374262401f-Paper.pdf
"""
from __future__ import absolute_import, division, print_function, unicode_literals, annotations

import time
from dataclasses import dataclass, replace

import tensorflow as tf
from tensorflow.keras.callbacks import Callback

from awp_protocol import batch_processor_no_proxy
from awp_protocol.attacks.v1 import pgd
from awp_protocol.attacks.attack import TensorflowEvasionAttack
from awp_protocol.callbacks.progbar_logger import ProgbarLogger
from awp_protocol.callbacks.checkpoint_callback import EpochCheckpoint

from awp_protocol.losses.loss import AdversarialLoss
from awp_protocol.losses.trades_loss import TradesLoss
from awp_protocol.losses.adversarial_categorical_cross_entropy import AdversarialSparseCategoricalCrossEntropy


@dataclass(frozen=True)
class Params:
    mode: str = "trades"
    protocol_params: batch_processor_no_proxy.AWPParams = batch_processor_no_proxy.AWPParams()

class AdversarialTrainerAWPTensorflow:
    """
    Class performing adversarial training following Adversarial Weight Perturbation (AWP) protocol.

    | Paper link: https://proceedings.neurips.cc/paper/2020/file/1ef91c212e30e14bf125e9374262401f-Paper.pdf
    """

    def __init__(
            self,
            classifier: tf.keras.Model,
            attack: TensorflowEvasionAttack,
            warmup: int = 0,
            adversarial_loss: AdversarialLoss | None = None,
            trained_layers: tuple[bool, ...] | None = None,
            params: Params | None = None,
            **overrides
    ):
        """
        Create an :class:`.AdversarialTrainerAWPPyTorch` instance.

        :param classifier: Model to train adversarially.
        :param proxy_classifier: Model for adversarial weight perturbation.
        :param attack: attack to use for data augmentation in adversarial training.
        :param mode: mode determining the optimization objective of base adversarial training and weight perturbation
               step
        :param gamma: The scaling factor controlling norm of weight perturbation relative to model parameters' norm.
        :param beta: The scaling factor controlling tradeoff between clean loss and adversarial loss for TRADES protocol
        :param warmup: The number of epochs after which weight perturbation is applied
        """
        self._fast_mode = True

        self._params = params or Params()
        self._params = replace(self._params, **overrides)

        self._classifier: tf.keras.Model = classifier
        self._attack: TensorflowEvasionAttack = attack
        self._warmup: int
        self._apply_wp: bool
        self._adversarial_loss: AdversarialLoss | None = adversarial_loss
        self._tracked_layers: tuple[bool, ...] | None = trained_layers

        self._steps_per_epoch: int | None = None
        self._epochs_run = 0
        self._trainer: batch_processor_no_proxy.BatchProcessor
        self._warmup = warmup

        self._progbar: tf.keras.utils.Progbar
        self._callback_list: tf.keras.callbacks.CallbackList
        self._logger: ProgbarLogger
        self._ckpt = EpochCheckpoint(self._classifier.name)

        self._clean_loss_metric = tf.keras.metrics.Mean()
        self._clean_accuracy_metric = tf.keras.metrics.SparseCategoricalAccuracy()
        self._robust_loss_metric = tf.keras.metrics.Mean()
        self._robust_accuracy_metric = tf.keras.metrics.SparseCategoricalAccuracy()


    def fit(
            self,
            x: tf.Tensor,
            y: tf.Tensor,
            validation_data: tuple[tf.Tensor, tf.Tensor] | None = None,
            batch_size: int = 128,
            nb_epochs: int = 1,
            callbacks: list[Callback] | None = None,
            enable_adversarial = True,
            **kwargs
    ):
        train_dataset = (
            tf.data.Dataset.from_tensor_slices((x, y))
            .batch(batch_size, drop_remainder=True)
            .prefetch(tf.data.AUTOTUNE))
        self._steps_per_epoch = train_dataset.cardinality().numpy() or None

        validation_dataset = None
        if validation_data:
            val_x, val_y = validation_data
            validation_dataset = (
                tf.data.Dataset.from_tensor_slices((val_x, val_y))
                .batch(batch_size, drop_remainder=True)
                .prefetch(tf.data.AUTOTUNE)
            )

        self._train_loop(train_dataset, nb_epochs, callbacks=callbacks, validation_dataset=validation_dataset, enable_adversarial=enable_adversarial)


    def fit_dataset(
            self,
            train_dataset: tf.data.Dataset,
            validation_dataset: tf.data.Dataset | None = None,
            nb_epochs: int = 1,
            callbacks: list[tf.keras.callbacks.Callback] | None = None,
            enable_adversarial = True,
            **kwargs
    ):
        self._steps_per_epoch = train_dataset.cardinality().numpy() or None
        self._train_loop(train_dataset, nb_epochs, callbacks=callbacks, validation_dataset=validation_dataset, enable_adversarial=enable_adversarial)


    def _train_loop(
            self,
            train_dataset,
            nb_epochs,
            validation_dataset=None,
            callbacks: list[tf.keras.callbacks.Callback] | None = None,
            steps_per_epoch: int = None,
            enable_adversarial=True
    ):
        callbacks = callbacks or []
        self._logger = ProgbarLogger()
        callbacks += [self._logger]
        self._callback_list = tf.keras.callbacks.CallbackList(callbacks, add_history=True, model=self._classifier)

        self._callback_list.on_train_begin()
        self._trainer = self._init_training_object()

        for epoch in range(nb_epochs):
            self._epoch(train_dataset, epoch + 1, validation_dataset=validation_dataset, enable_adversarial=enable_adversarial)

        self._callback_list.on_train_end()


    def _epoch(self, train_dataset: tf.data.Dataset, epoch: int, validation_dataset: tf.data.Dataset | None = None, enable_adversarial=True):
        self._reset_metrics()

        self._progbar = tf.keras.utils.Progbar(
            self._steps_per_epoch,
            stateful_metrics=["loss", "accuracy", "robust_loss", "robust_accuracy"],
        )
        self._logger.update_progbar(self._progbar)

        self._callback_list.on_epoch_begin(self._epochs_run)

        start_time = time.time()
        warmup = epoch <= self._warmup
        for step, (x_batch, y_batch) in enumerate(train_dataset):
            self._run_batch(x_batch, y_batch, step+1, warmup=warmup, enable_adversarial=enable_adversarial)
        end_time = time.time()
        train_time = end_time - start_time

        logs = self._collect_train_logs()
        lr = None
        if self._classifier.optimizer is not None:
            lr = self._classifier.optimizer.learning_rate
        logs.update({
            "train_time": train_time,
            "lr": lr,
        })

        if validation_dataset is not None:
            self._reset_metrics()
            start_time = time.time()
            self._run_validation(validation_dataset)
            end_time = time.time()
            validation_time = end_time - start_time
            logs.update({
                "val_loss": self._clean_loss_metric.result(),
                "val_accuracy": self._clean_accuracy_metric.result(),
                "val_robust_loss": self._robust_loss_metric.result(),
                "val_robust_accuracy": self._robust_accuracy_metric.result(),
                "val_time": validation_time,
            })

        self._progbar.update(self._steps_per_epoch, finalize=True)
        self._callback_list.on_epoch_end(epoch, logs)


    def _run_batch(self, x_batch: tf.Tensor, y_batch: tf.Tensor, step, warmup, enable_adversarial=True):
        self._callback_list.on_batch_begin(step)

        batch_results = self._train_step(x_batch, y_batch, warmup=warmup, enable_adversarial=enable_adversarial)
        self._update_metrics(y_batch, batch_results)
        self._callback_list.on_batch_end(step, self._collect_train_logs())

    def _collect_train_logs(self):
        logs = {
            "loss": self._clean_loss_metric.result(),
            "accuracy": self._clean_accuracy_metric.result(),
            "robust_loss": self._robust_loss_metric.result(),
            "robust_accuracy": self._robust_accuracy_metric.result(),
        }
        return logs

    def _run_validation(self, validation_dataset):
        for x_batch, y_batch in validation_dataset:
            batch_results = self._trainer.validation_step(x_batch, y_batch)
            self._update_metrics(y_batch, batch_results)


    def _update_metrics(self, y_batch, batch_results: tuple):
        clean_loss, clean_logits, robust_loss, robust_logits = batch_results
        self._clean_loss_metric.update_state(clean_loss)
        self._clean_accuracy_metric.update_state(y_batch, clean_logits)
        self._robust_loss_metric.update_state(robust_loss)
        self._robust_accuracy_metric.update_state(y_batch, robust_logits)


    def _reset_metrics(self):
        self._clean_loss_metric.reset_state()
        self._clean_accuracy_metric.reset_state()
        self._robust_loss_metric.reset_state()
        self._robust_accuracy_metric.reset_state()


    def _train_step(self, x_batch: tf.Tensor, y_batch: tf.Tensor, warmup: bool, enable_adversarial=True) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        if not enable_adversarial:
            return self._non_adversarial_step(x_batch, y_batch)
        if warmup:
            return self._trainer.adv_train_step(x_batch, y_batch)
        else:
            return self._trainer.awp_train_step(x_batch, y_batch)


    @tf.function
    def _non_adversarial_step(self, x_batch, y_batch) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        with tf.GradientTape() as tape:
            logits = self._classifier(x_batch)
            loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True)(y_batch, logits)
        gradient = tape.gradient(loss, self._classifier.trainable_variables)
        self._classifier.optimizer.apply_gradients(zip(gradient, self._classifier.trainable_variables))
        return loss, logits, loss, logits


    def _init_training_object(self):
        attack = self._attack or pgd.PGDAttack(self._classifier)
        adversarial_loss = self._adversarial_loss or _select_adversarial_loss(self._params.mode)
        tracked_layers = self._tracked_layers or select_default_trained_layers_tf(self._classifier)

        return batch_processor_no_proxy.BatchProcessor(
            self._classifier,
            attack,
            adversarial_loss,
            tracked_layers=tracked_layers,
            params=self._params.protocol_params
        )


def select_default_trained_layers_tf(classifier: tf.keras.Model) -> tuple[bool, ...]:
        return tuple('kernel' in variable.name for variable in classifier.trainable_variables)


def _select_adversarial_loss(mode: str) -> AdversarialLoss:
    if mode == "pgd":
        return AdversarialSparseCategoricalCrossEntropy()
    if mode == "trades":
        return TradesLoss()
    else:
        raise Exception("Mode not provided! Chose pgd or trades.")

