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
from dataclasses import dataclass, replace

import logging

import tensorflow as tf
from scipy.linalg import _decomp_update
from tensorflow.keras.callbacks import Callback
from tensorflow.python.tools.api.generator2.generator import generator

import batch_processor
from awp_protocol.attacks import pgd
from awp_protocol.attacks.attack import TensorflowEvasionAttack
from callbacks.progbar_logger import ProgbarLogger
from callbacks.checkpoint_callback import EpochCheckpoint

logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class Params:
    protocol_params: batch_processor.AWPParams = batch_processor.AWPParams()

class AdversarialTrainerAWPTensorflow:
    """
    Class performing adversarial training following Adversarial Weight Perturbation (AWP) protocol.

    | Paper link: https://proceedings.neurips.cc/paper/2020/file/1ef91c212e30e14bf125e9374262401f-Paper.pdf
    """

    def __init__(
            self,
            classifier: tf.keras.Model,
            proxy_classifier: tf.keras.Model,
            attack: TensorflowEvasionAttack,
            warmup: int = 0,
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
        self._proxy_classifier: tf.keras.Model = proxy_classifier
        self._attack: TensorflowEvasionAttack = attack
        self._warmup: int
        self._apply_wp: bool
        self._tracked_layers: tuple[bool, ...] | None = trained_layers

        self._steps_per_epoch: int | None = None
        self._epochs_run = 0
        self._trainer: batch_processor.BatchProcessor
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
            **kwargs,
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

        self._train_loop(train_dataset, nb_epochs, callbacks=callbacks, validation_dataset=validation_dataset)


    def fit_dataset(
            self,
            train_dataset: tf.data.Dataset,
            validation_dataset: tf.data.Dataset | None = None,
            nb_epochs: int = 1,
            callbacks: list[tf.keras.callbacks.Callback] | None = None
    ):
        self._steps_per_epoch = train_dataset.cardinality().numpy() or None
        self._train_loop(train_dataset, nb_epochs, callbacks=callbacks, validation_dataset=validation_dataset)


    def _train_loop(
            self,
            train_dataset,
            nb_epochs,
            validation_dataset=None,
            callbacks: list[tf.keras.callbacks.Callback] | None = None,
            steps_per_epoch: int = None,
    ):
        callbacks = callbacks or []
        self._logger = ProgbarLogger()
        callbacks += [self._logger]
        self._callback_list = tf.keras.callbacks.CallbackList(callbacks, add_history=True, model=self._classifier)

        self._callback_list.on_train_begin()
        self._trainer = self._init_training_object()

        logger.info("Performing adversarial training with AWP with %s protocol", self._params.protocol_params.mode)

        for epoch in range(nb_epochs):
            self._epoch(train_dataset, validation_dataset)

        self._callback_list.on_train_end()


    def _epoch(self, train_dataset: tf.data.Dataset, validation_dataset: tf.data.Dataset | None = None):
        self._epochs_run += 1
        self._reset_metrics()

        self._progbar = tf.keras.utils.Progbar(
            self._steps_per_epoch,
            stateful_metrics=["loss", "accuracy"]
        )
        self._logger.update_progbar(self._progbar)

        self._callback_list.on_epoch_begin(self._epochs_run)

        for step, (x_batch, y_batch) in enumerate(train_dataset):
            self._run_batch(x_batch, y_batch, step)

        logs = {
            "loss": self._clean_loss_metric.result(),
            "accuracy": self._clean_accuracy_metric.result(),
            "robust_loss": self._robust_loss_metric.result(),
            "robust_accuracy": self._robust_accuracy_metric.result(),
        }

        if validation_dataset is not None:
            self._run_validation(validation_dataset)
            logs.update({
                "val_loss": self._clean_loss_metric.result(),
                "val_accuracy": self._clean_accuracy_metric.result(),
                "robust_loss": self._robust_loss_metric.result(),
                "robust_accuracy": self._robust_accuracy_metric.result(),
            })

        self._callback_list.on_epoch_end(self._epochs_run, logs)


    def _run_batch(self, x_batch: tf.Tensor, y_batch: tf.Tensor, step):
        self._callback_list.on_batch_begin(step)

        warmup = self._epochs_run < self._warmup
        batch_results = self._train_step(x_batch, y_batch, warmup=warmup)
        self._update_metrics(y_batch, batch_results)

        self._callback_list.on_batch_end(self._epochs_run, {"loss": self._clean_loss_metric.result()})


    def _run_validation(self, validation_dataset):
        self._reset_metrics()
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


    def _train_step(self, x_batch: tf.Tensor, y_batch: tf.Tensor, warmup: bool) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        if warmup:
            return self._trainer.adv_train_step(x_batch, y_batch)
        else:
            return self._trainer.awp_train_step(x_batch, y_batch)


    def _init_training_object(self):
        attack = pgd.PGDAttack(self._proxy_classifier)
        tracked_layers = self._tracked_layers or select_default_trained_layers_tf(self._proxy_classifier)
        return batch_processor.BatchProcessor(
            self._classifier,
            self._proxy_classifier,
            tracked_layers,
            attack,
            self._params.protocol_params)


def clone_classifier(originator: tf.keras.Model) -> tf.keras.Model:
    proxy_classifier = tf.keras.models.clone_model(originator)
    proxy_classifier.set_weights(originator.get_weights())
    return proxy_classifier

def select_default_trained_layers_tf(classifier: tf.keras.Model) -> tuple[bool, ...]:
        return tuple('kernel' in variable.name for variable in classifier.trainable_variables)
