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

import numpy as np

from art.defences.trainer.adversarial_trainer_awp_pytorch import AdversarialTrainerAWP, AdversarialTrainerAWPPyTorch
from art.estimators.classification.tensorflow import TensorFlowV2Classifier

import tensorflow as tf
from tensorflow.keras.callbacks import Callback
from tensorflow.python.tools.api.generator2.generator import generator

from awp_protocol import awp_protocol_tf, awp_proxy
from awp_protocol.attacks import pgd
from awp_protocol.attacks.attack import TensorflowEvasionAttack
from awp_protocol.losses.loss_context import LossContext

logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class AWPParams:
    protocol_params: awp_protocol_tf.AWPProtocolParams = awp_protocol_tf.AWPProtocolParams()

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
            params: AWPParams | None = None,
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
        self._classifier: tf.keras.Model = classifier
        self._proxy_classifier: tf.keras.Model = proxy_classifier
        self._attack: TensorflowEvasionAttack = attack
        self._warmup: int
        self._apply_wp: bool
        self._tracked_layers: tuple[bool, ...] | None = trained_layers
        self._params = params or AWPParams()
        self._params = replace(self._params, **overrides)

        self._loss_metric = tf.keras.metrics.Mean()
        self._accuracy_metric = tf.keras.metrics.SparseCategoricalAccuracy()
        self._steps_per_epoch = None
        self._epochs_run = 0
        self._callback_list: tf.keras.callbacks.CallbackList
        self._progbar: tf.keras.utils.Progbar
        self._trainer: awp_protocol_tf.AWPProtocolTF
        self._warmup = warmup

        self._fast_mode = True

    def fit(
            self,
            x: np.ndarray,
            y: np.ndarray,
            validation_data: tuple[np.ndarray, np.ndarray] | None = None,
            batch_size: int = 128,
            nb_epochs: int = 1,
            callbacks: list[Callback] | None = None,
            **kwargs,
    ):
        """
        Train model with AWP protocol.
        See class documentation for more information on the exact procedure.

        :param x: Training set.
        :param y: Labels for the training set.
        :param validation_data: Tuple consisting of validation data, (x_val, y_val)
        :param batch_size: Size of batches.
        :param nb_epochs: Number of epochs to use for trainings.
        :param callbacks: List of callbacks as in keras
        :param kwargs: Dictionary of framework-specific arguments. These will be passed as such to the `fit` function of
                                  the target classifier.
        """

        iterator_fn = lambda: self._numpy_to_iterator(x, y, batch_size)

        validation_fn = None
        if validation_data:
            validation_fn = lambda: self._validate_dataset(*validation_data)

        self._train_loop(iterator_fn, nb_epochs, callbacks, validation_fn)

    def _transform_dataset(self, dataset: tf.data.Dataset, nb_classes: int, apply_fit: bool):
        def check_transform_and_preprocess(x, y):
            y = tf.squeeze(y)
            y = tf.cast(y, tf.int32)
            x, y = self._classifier._apply_preprocessing(x, y, fit=apply_fit)
            return x, y
        transformed_dataset = dataset.map(check_transform_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
        return transformed_dataset

    def fit_dataset(self,
            train_dataset: tf.data.Dataset,
            validation_dataset: tf.data.Dataset | None = None,
            nb_epochs: int = 1,
            callbacks: list[tf.keras.callbacks.Callback] | None = None,
            **kwargs):

        """
                Train model with AWP protocol.
                See class documentation for more information on the exact procedure.

                :param x: Training set.
                :param y: Labels for the training set.
                :param validation_data: Tuple consisting of validation data, (x_val, y_val)
                :param batch_size: Size of batches.
                :param nb_epochs: Number of epochs to use for trainings.
                :param callbacks: List of callbacks as in keras
                :param kwargs: Dictionary of framework-specific arguments. These will be passed as such to the `fit` function of
                                          the target classifier.
                """

        steps_per_epoch=train_dataset.cardinality()
        self._train_loop(train_dataset, nb_epochs, callbacks=callbacks, validation_dataset=validation_dataset, steps_per_epoch=steps_per_epoch)

    def _train_loop(
            self,
            train_dataset,
            nb_epochs,
            validation_dataset=None,
            callbacks: list[tf.keras.callbacks.Callback] | None = None,
            steps_per_epoch: int = None,
    ):
        callbacks = callbacks or []
        self._callback_list = tf.keras.callbacks.CallbackList(callbacks, add_history=True, model=self._classifier)
        logs = {"loss": None, "accuracy": None}
        self._callback_list.on_train_begin()
        self._trainer = self._init_training_object()

        logger.info("Performing adversarial training with AWP with %s protocol", self._params.protocol_params.mode)
        for epoch in range(nb_epochs):
            self._epoch_step(train_dataset, validation_dataset)

        self._callback_list.on_train_end(logs)

    def _epoch_step(self, dataset, validation_fn=None):
        self._epochs_run += 1
        self._callback_list.on_epoch_begin(self._epochs_run)
        self._loss_metric.reset_state()
        self._accuracy_metric.reset_state()

        if hasattr(dataset, "__len__"):
            total_steps = len(dataset)
        else:
            total_steps = self._steps_per_epoch

        self._progbar = tf.keras.utils.Progbar(
            total_steps,
            stateful_metrics=["loss", "accuracy"]
        )

        step = 0
        for step, (x_batch, y_batch) in enumerate(dataset):
            self._run_batch(x_batch, y_batch, step)

        if self._steps_per_epoch is None:
            self._steps_per_epoch = step + 1
        logs = {
            "loss": self._loss_metric.result(),
            "accuracy": self._accuracy_metric.result()
        }
        if validation_fn:
            logs.update(validation_fn())
        self._callback_list.on_epoch_end(self._epochs_run, logs)


    def _run_batch(self, x_batch: tf.Tensor, y_batch: tf.Tensor, step):
        if not self._fast_mode:
            self._callback_list.on_batch_begin(step)

        warmup = self._epochs_run < self._warmup
        self._train_step(x_batch, y_batch, warmup=warmup)

        if not self._fast_mode:
            self._callback_list.on_batch_end(self._epochs_run, {"loss": self._loss_metric.result()})

        if step % 10 == 0:
            if not self._fast_mode:
                values = [("loss", self._loss_metric.result()), ("accuracy", self._accuracy_metric.result())]
                self._progbar.update(step + 1, values=values)
            else:
                values = [("loss", self._loss_metric.result())]
                self._progbar.update(step + 1, values=values)

    @tf.function(jit_compile=False)
    def _train_step(self, x_batch: tf.Tensor, y_batch: tf.Tensor, warmup: bool):
        if warmup:
            loss, logits =  self._warmup_step(x_batch, y_batch)
        else:
            loss, logits =  self._trainer.batch_process(x_batch, y_batch)
        self._loss_metric.update_state(loss)
        if not self._fast_mode:
            self._accuracy_metric.update_state(y_batch, logits)

    def _warmup_step(self, x_batch: tf.Tensor, y_batch: tf.Tensor):
        with tf.GradientTape() as tape:
            logits = self._classifier(x_batch, training=True)
            loss = self._classifier.loss(y_batch, logits)
        grad = tape.gradient(loss, self._classifier.trainable_variables)
        self._classifier.optimizer.apply_gradients(zip(grad, self._classifier.trainable_variables))
        return loss, logits

    def run_validation(self, validation_dataset):
        if validation_dataset is None:
            return {}

        clean_loss = tf.keras.metrics.Mean()
        adv_loss = tf.keras.metrics.Mean()

        clean_acc = tf.keras.metrics.SparseCategoricalAccuracy()
        adv_acc = tf.keras.metrics.SparseCategoricalAccuracy()

        validation_dataset = validation_dataset.prefetch(tf.data.AUTOTUNE)

        for x_batch, y_batch in validation_dataset:
            y_pred = self._classifier(x_batch, training=False)
            loss_clean = tf.reduce_mean(self._classifier.loss(y_batch, y_pred))

            clean_loss.update_state(loss_clean)
            clean_acc.update_state(y_batch, y_pred)

            # -----------------------
            # ADVERSARIAL
            # -----------------------
            x_adv = self._attack.generate(x_batch, y_batch)

            y_adv_pred = self._classifier(x_adv, training=False)

            loss_adv = tf.reduce_mean(self._classifier.loss(y_batch, y_adv_pred))

            adv_loss.update_state(loss_adv)
            adv_acc.update_state(y_batch, y_adv_pred)

        return {
            "val_loss": clean_loss.result(),
            "val_adv_loss": adv_loss.result(),
            "val_acc": clean_acc.result(),
            "val_adv_acc": adv_acc.result(),
        }

    def _dataset_to_iterator(self, dataset):
        for x_batch, y_batch in dataset:
            yield x_batch, y_batch

    def _numpy_to_iterator(self, x, y, batch_size):
        n = len(x)
        indices = np.arange(n)
        np.random.shuffle(indices)

        for i in range(0, n, batch_size):
            batch_idx = indices[i:i + batch_size]
            yield x[batch_idx], y[batch_idx]

    def _generator_to_iterator(self, generator, nb_batches):
        for _ in range(nb_batches):
            x_batch, y_batch = generator.get_batch()
            yield x_batch, y_batch

    def _init_training_object(self):
        attack = pgd.PGDAttack(self._proxy_classifier)
        tracked_layers = self._tracked_layers or select_default_trained_layers_tf(self._proxy_classifier)
        return awp_protocol_tf.AWPProtocolTF(
            self._classifier,
            self._proxy_classifier,
            tracked_layers,
            attack,
            self._params.protocol_params)

    def _create_proxy_calculation_object(self) -> awp_protocol_tf.AWPProxyCalculations:
        tracked_layers = self._tracked_layers or select_default_trained_layers_tf(self._proxy_classifier)
        return awp_protocol_tf.AWPProxyCalculations(self._proxy_classifier, tracked_layers, self._params.awp_params)



def clone_classifier(originator: tf.keras.Model) -> tf.keras.Model:
    proxy_classifier = tf.keras.models.clone_model(originator)
    proxy_classifier.set_weights(originator.get_weights())
    return proxy_classifier

def select_default_trained_layers_tf(classifier: tf.keras.Model) -> tuple[bool, ...]:
        return tuple('kernel' in variable.name for variable in classifier.trainable_variables)
