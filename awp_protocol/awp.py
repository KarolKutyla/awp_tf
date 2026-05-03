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

import keras
import numpy as np
from tqdm.auto import tqdm, trange

from art.defences.trainer.adversarial_trainer_awp_pytorch import AdversarialTrainerAWP, AdversarialTrainerAWPPyTorch
from art.estimators.classification.tensorflow import TensorFlowV2Classifier
from art.data_generators import DataGenerator
from art.attacks.attack import EvasionAttack
from art.utils import check_and_transform_label_format

import tensorflow as tf
from tensorflow.keras.callbacks import Callback

from awp_protocol import awp_protocol_tf, awp_proxy
from awp_protocol.attacks import pgd
from awp_protocol.attacks.attack import TensorflowEvasionAttack
from awp_protocol.losses.loss_context import LossContext

logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class AWPParams:
    protocol_params = awp_protocol_tf.AWPProtocolParams()
    awp_params = awp_proxy.AWPProxyParams()

class AdversarialTrainerAWPTensorflow(AdversarialTrainerAWP):
    """
    Class performing adversarial training following Adversarial Weight Perturbation (AWP) protocol.

    | Paper link: https://proceedings.neurips.cc/paper/2020/file/1ef91c212e30e14bf125e9374262401f-Paper.pdf
    """

    def __init__(
            self,
            classifier: TensorFlowV2Classifier,
            proxy_classifier: TensorFlowV2Classifier,
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
        super().__init__(classifier, proxy_classifier, None, None, None, None, warmup)
        self._classifier: TensorFlowV2Classifier = classifier
        self._proxy_classifier: TensorFlowV2Classifier = proxy_classifier
        self._attack: EvasionAttack = attack
        self._warmup: int
        self._apply_wp: bool
        self._tracked_layers: tuple[bool, ...] | None = trained_layers
        self._params = params or AWPParams()
        self._params = replace(self._params, **overrides)

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

        iterator_fn = lambda: self._dataset_to_iterator(train_dataset)

        validation_fn = None
        if validation_dataset:
            validation_dataset = self._transform_dataset(validation_dataset, self.classifier.nb_classes, False)
            validation_fn = lambda: self.run_validation(validation_dataset)

        self._train_loop(iterator_fn, nb_epochs, callbacks, validation_fn)


    def run_validation(self, validation_dataset):
        if validation_dataset is None:
            return {}

        clean_loss = tf.keras.metrics.Mean()
        adv_loss = tf.keras.metrics.Mean()

        clean_acc = tf.keras.metrics.SparseCategoricalAccuracy()
        adv_acc = tf.keras.metrics.SparseCategoricalAccuracy()

        validation_dataset = validation_dataset.prefetch(tf.data.AUTOTUNE)

        for x_batch, y_batch in validation_dataset:
            y_pred = self.classifier.model(x_batch, training=False)
            loss_clean = tf.reduce_mean(self.classifier.model.loss_object(y_batch, y_pred))

            clean_loss.update_state(loss_clean)
            clean_acc.update_state(y_batch, y_pred)

            # -----------------------
            # ADVERSARIAL
            # -----------------------
            x_adv = self._attack.generate(x_batch, y_batch)

            y_adv_pred = self.classifier.model(x_adv, training=False)

            loss_adv = tf.reduce_mean(self.classifier.model.loss_object(y_batch, y_adv_pred))

            adv_loss.update_state(loss_adv)
            adv_acc.update_state(y_batch, y_adv_pred)

        return {
            "val_loss": clean_loss.result(),
            "val_adv_loss": adv_loss.result(),
            "val_acc": clean_acc.result(),
            "val_adv_acc": adv_acc.result(),
        }

    def fit_generator(
            self,
            generator: DataGenerator,
            validation_data: tuple[np.ndarray, np.ndarray] | None = None,
            nb_epochs: int = 20,
            scheduler: None = None,
            callbacks: Callback | None = None,
            warmup: int = 0,
            **kwargs,
    ):
        """
        Train model with AWP protocol using a data generator.
        See class documentation for more information on the exact procedure.

        :param generator: Data generator.
        :param validation_data: Tuple consisting of validation data, (x_val, y_val)
        :param nb_epochs: Number of epochs to use for trainings.
        :param scheduler: Learning rate scheduler to run at the end of every epoch.
        :param kwargs: Dictionary of framework-specific arguments. These will be passed as such to the `fit` function of
                                  the target classifier.
        """

        nb_batches = int(np.ceil(generator.size / generator.batch_size))

        iterator_fn = lambda: self._generator_to_iterator(generator, nb_batches)

        validation_fn = None
        if validation_data:
            validation_fn = lambda: self._validate_dataset(*validation_data)

        self._train_loop(iterator_fn, nb_epochs, callbacks, validation_fn, warmup=warmup)

    def _train_loop(
            self,
            iterator_fn,
            nb_epochs,
            callbacks: list[tf.keras.callbacks.Callback] | None = None,
            validation_fn=None,
            warmup: int = 0,
            steps_per_epoch: int | None = None,
    ):
        logger.info("Performing adversarial training with AWP with %s protocol", self._mode)
        callbacks = callbacks or []
        callback_list = tf.keras.callbacks.CallbackList(callbacks, add_history=True, model=self.classifier.model)
        logs = {}

        callback_list.on_train_begin()

        trainer = self._init_training_object()

        for epoch in range(nb_epochs):

            callback_list.on_epoch_begin(epoch)

            epoch_logs = {
            "train_loss": 0.0,
            "train_acc": 0.0,
            "train_n": 0.0
            }

            iterator = iterator_fn()
            if hasattr(iterator, "__len__"):
                total_steps = len(iterator)
            else:
                total_steps = steps_per_epoch

            progbar = tf.keras.utils.Progbar(
                total_steps,
                stateful_metrics=["loss", "acc"]
            )

            for step, (x_batch, y_batch) in enumerate(iterator):

                callback_list.on_batch_begin(step)

                if epoch >= self._warmup:
                    logs = self._adv_step(x_batch, y_batch, trainer, epoch_logs)
                else:
                    logs = self._warmup_step(x_batch, y_batch, epoch_logs)

                loss = logs["loss"]
                acc = logs.get("acc", None)
                values = [("loss", float(loss))]
                if acc is not None:
                    values.append(("acc", float(acc)))
                progbar.update(step + 1, values=values)

                callback_list.on_batch_end(step, logs)

                # postfix = {"loss": float(logs["loss"])}
                # if "acc" in logs.keys():
                #     postfix["acc"] = float(logs["acc"])
                # pbar.set_postfix(postfix)

            logs = {
                "loss": epoch_logs["train_loss"] / epoch_logs["train_n"],
                "acc": epoch_logs["train_acc"] / epoch_logs["train_n"]
            }

            if validation_fn:
                logs.update(validation_fn())
            callback_list.on_epoch_end(epoch, logs)

        callback_list.on_train_end(logs)

    def _warmup_step(self, x_batch, y_batch, epoch_dict):
        with tf.GradientTape() as tape:
            logits = self.classifier.model(x_batch, training=True)
            loss = self.classifier.model.loss(y_batch, logits)
        grad = tape.gradient(loss, self.classifier.model.trainable_variables)
        self.classifier.model.optimizer.apply_gradients(zip(grad, self.classifier.model.trainable_variables))
        batch_size = tf.shape(x_batch)[0]
        epoch_dict["train_n"]  += tf.cast(batch_size, tf.float32)
        return {
            "loss": float(loss.numpy()) if tf.is_tensor(loss) else loss
        }

    def _adv_step(self, x_batch, y_batch, trainer, epoch_dict):
        metrics = trainer.batch_process(x_batch, y_batch)
        loss = metrics['loss']
        ctx: LossContext = metrics['ctx']
        accuracy = tf.reduce_mean(
            tf.keras.metrics.sparse_categorical_accuracy(y_batch, ctx.logits_pert)
        )

        batch_size = tf.shape(x_batch)[0]

        epoch_dict["train_loss"] += loss * tf.cast(batch_size, tf.float32)
        epoch_dict["train_acc"]  += accuracy * tf.cast(batch_size, tf.float32)
        epoch_dict["train_n"]  += tf.cast(batch_size, tf.float32)

        return {
            "loss": float(loss.numpy()) if tf.is_tensor(loss) else loss,
            "acc": float(accuracy.numpy()) if tf.is_tensor(accuracy) else accuracy
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
        attack = pgd.PGDAttack(self._proxy_classifier.model)
        return awp_protocol_tf.AWPProtocolTF(
            self._classifier.model,
            self._create_proxy_calculation_object(),
            attack,
            self._classifier.optimizer,
            self._params.protocol_params)

    def _create_proxy_calculation_object(self) -> awp_protocol_tf.AWPProxyCalculations:
        tracked_layers = self._tracked_layers or select_default_trained_layers_tf(self._proxy_classifier.model)
        return awp_protocol_tf.AWPProxyCalculations(self._proxy_classifier.model, tracked_layers, self._params.awp_params)



def clone_classifier(originator: tf.keras.Model) -> tf.keras.Model:
    proxy_classifier = tf.keras.models.clone_model(originator)
    proxy_classifier.set_weights(originator.get_weights())
    return proxy_classifier

def select_default_trained_layers_tf(classifier) -> tuple[bool, ...]:
        return tuple('kernel' in variable.name for variable in classifier.trainable_variables)
