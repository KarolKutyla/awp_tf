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

import copy
import logging
import time
from typing import TYPE_CHECKING

from collections import OrderedDict
import numpy as np
from tqdm.auto import trange

from art.defences.trainer.adversarial_trainer_awp_pytorch import AdversarialTrainerAWP, AdversarialTrainerAWPPyTorch
from art.estimators.classification.tensorflow import TensorFlowV2Classifier
from art.data_generators import DataGenerator
from art.attacks.attack import EvasionAttack
from art.utils import check_and_transform_label_format
from tensorflow.data import Dataset
import tensorflow as tf
from neural_network_analytic_tool.art_tf.awp_protocol_tf import AWPProtocolTF
from tensorflow import keras
from keras.callbacks import Callback
from art.utils import projection

logger = logging.getLogger(__name__)
EPS = 1e-8  # small value required for avoiding division by zero and for KLDivLoss to make probability vector non-zero
EPS_2 = 32 / 255


class AdversarialTrainerAWPTensorflow(AdversarialTrainerAWP):
    """
    Class performing adversarial training following Adversarial Weight Perturbation (AWP) protocol.

    | Paper link: https://proceedings.neurips.cc/paper/2020/file/1ef91c212e30e14bf125e9374262401f-Paper.pdf
    """

    def __init__(
            self,
            classifier: TensorFlowV2Classifier,
            proxy_classifier: TensorFlowV2Classifier,
            attack: EvasionAttack | None,
            mode: str,
            gamma: float,
            beta: float,
            warmup: int,
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
        super().__init__(classifier, proxy_classifier, attack, mode, gamma, beta, warmup)
        self._classifier: TensorFlowV2Classifier
        self._proxy_classifier: TensorFlowV2Classifier
        self._attack: EvasionAttack
        self._mode: str
        self.gamma: float
        self._beta: float
        self._warmup: int
        self._apply_wp: bool

    def fit(
            self,
            x: np.ndarray,
            y: np.ndarray,
            validation_data: tuple[np.ndarray, np.ndarray] | None = None,
            batch_size: int = 128,
            nb_epochs: int = 20,
            callbacks: list[Callback] = [],
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

        logger.info("Performing adversarial training with AWP with %s protocol", self._mode)
        for callback in callbacks:
            callback.on_train_begin()

        best_acc_adv_test = 0
        nb_batches = int(np.ceil(len(x) / batch_size))
        ind = np.arange(len(x))

        logger.info("Adversarial Training AWP with %s", self._mode)
        y = check_and_transform_label_format(y, nb_classes=self.classifier.nb_classes)
        # adversarial_perturbed_weights = self.AWPProtocolTF(self._classifier, self._classifier.loss_object,
        #                                                    attack=self._attack, optimizer=self._classifier.optimizer)
        awp_protocol_loop_tf = AWPProtocolTF(self._classifier.model, self._classifier.loss_object, optimizer=self._classifier.optimizer, attack=self._attack)

        for i_epoch in trange(nb_epochs, desc=f"Adversarial Training AWP with {self._mode} - Epochs"):
            for callback in callbacks:
                callback.on_epoch_begin(i_epoch)
            if i_epoch >= self._warmup:
                self._apply_wp = True

            # Shuffle the examples
            np.random.shuffle(ind)
            start_time = time.time()
            train_loss = 0.0
            train_acc = 0.0
            train_n = 0.0

            for batch_id in range(nb_batches):
                # Create batch data
                for callback in callbacks:
                    callback.on_batch_begin(batch_id)
                x_batch = x[ind[batch_id * batch_size: min((batch_id + 1) * batch_size, x.shape[0])]].copy()
                y_batch = y[ind[batch_id * batch_size: min((batch_id + 1) * batch_size, x.shape[0])]]

                _train_loss, _train_acc, _train_n = self._batch_process(x_batch, y_batch, awp_protocol_loop_tf)

                train_loss += _train_loss
                train_acc += _train_acc
                train_n += _train_n

                for callback in callbacks:
                    logs = {"loss": _train_loss, "acc": _train_acc}
                    callback.on_batch_end(i_epoch, logs)

            train_time = time.time()
            # compute accuracy
            if validation_data is not None:
                (x_test, y_test) = validation_data
                y_test = check_and_transform_label_format(y_test, nb_classes=self.classifier.nb_classes)

                x_preprocessed_test, y_preprocessed_test = self._classifier._apply_preprocessing(
                    x_test,
                    y_test,
                    fit=True,
                )
                # pylint: enable=protected-access
                output_clean = np.argmax(self.predict(x_preprocessed_test), axis=1)
                nb_correct_clean = np.sum(output_clean == np.argmax(y_preprocessed_test, axis=1))
                x_test_adv = self._attack.generate(x_preprocessed_test, y=y_preprocessed_test)
                output_adv = np.argmax(self.predict(x_test_adv), axis=1)
                nb_correct_adv = np.sum(output_adv == np.argmax(y_preprocessed_test, axis=1))

                logger.info(
                    "epoch: %s time(s): %.1f loss: %.4f acc-adv (tr): %.4f acc-clean (val): %.4f acc-adv (val): %.4f",
                    i_epoch,
                    train_time - start_time,
                    train_loss / train_n,
                    train_acc / train_n,
                    nb_correct_clean / x_test.shape[0],
                    nb_correct_adv / x_test.shape[0],
                )

                # save last checkpoint
                if i_epoch + 1 == nb_epochs:
                    self._classifier.save(filename=f"awp_{self._mode.lower()}_epoch_{i_epoch}")

                # save best checkpoint
                if nb_correct_adv / x_test.shape[0] > best_acc_adv_test:
                    self._classifier.save(filename=f"awp_{self._mode.lower()}_epoch_best")
                    best_acc_adv_test = nb_correct_adv / x_test.shape[0]

            else:
                logger.info(
                    "epoch: %s time(s): %.1f loss: %.4f acc-adv: %.4f",
                    i_epoch,
                    train_time - start_time,
                    train_loss / train_n,
                    train_acc / train_n,
                )
            for callback in callbacks:
                logs = {"loss": train_loss, "acc": train_acc}
                callback.on_epoch_end(i_epoch, logs)

        for callback in callbacks:
            logs = {"loss": train_loss, "acc": train_acc}
            callback.on_train_end(logs)

    def _transform_dataset(self, dataset: tf.data.Dataset, nb_classes: int, apply_fit: bool):
        from copy import copy
        def check_transform_and_preprocess(x, y):
            y = check_and_transform_label_format(y, nb_classes)
            x, y = self._classifier._apply_preprocessing(x, y, fit=apply_fit)
            return x, y
        transformed_dataset = copy(dataset)
        transformed_dataset.map(check_transform_and_preprocess)
        return transformed_dataset

    def fit_dataset(self,
            train_dataset: Dataset,
            validation_dataset: Dataset | None = None,
            nb_epochs: int = 20,
            callbacks: list[Callback] = [],
            **kwargs,):

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

        logger.info("Performing adversarial training with AWP with %s protocol", self._mode)
        nb_batches = tf.data.experimental.cardinality(train_dataset).numpy()
        train_dataset = self._transform_dataset(train_dataset, self.classifier.nb_classes, apply_fit=True)
        if validation_dataset is not None:
            validation_dataset = self._transform_dataset(validation_dataset, self.classifier.nb_classes, apply_fit=False)

        for callback in callbacks:
            params = {"steps": nb_batches, "epochs": nb_epochs}
            callback.set_model(self.classifier)
            callback.set_params(params)
            callback.on_train_begin()

        best_acc_adv_test = 0

        logger.info("Adversarial Training AWP with %s", self._mode)
        adversarial_perturbed_weights = AWPProtocolTF(self._classifier.model, self._classifier.loss_object,
                                                           optimizer=self._classifier.optimizer, attack=self._attack)

        for i_epoch in trange(nb_epochs, desc=f"Adversarial Training AWP with {self._mode} - Epochs"):
            for callback in callbacks:
                callback.on_epoch_begin(i_epoch)
            if i_epoch >= self._warmup:
                self._apply_wp = True

            # Shuffle the examples
            start_time = time.time()
            train_loss = 0.0
            train_acc = 0.0
            train_n = 0.0

            for batch_id, batch in enumerate(train_dataset.as_numpy_iterator()):
                # Create batch data
                for callback in callbacks:
                    callback.on_batch_begin(batch_id)

                x_batch = batch[0].copy()
                y_batch = batch[1]

                _train_loss, _train_acc, _train_n = self._batch_process(x_batch, y_batch, adversarial_perturbed_weights)

                train_loss += _train_loss
                train_acc += _train_acc
                train_n += _train_n

                for callback in callbacks:
                    logs = {"loss": _train_loss, "acc": _train_acc}
                    callback.on_batch_end(i_epoch, logs)

            train_time = time.time()
            # compute accuracy
            if validation_dataset is not None:
                sum_batches = 0
                normal_loss_sum = 0.0
                adv_loss_sum = 0.0
                labels_list = []
                labels_clean_list = []
                labels_adv_list = []
                for batch_id, batch in enumerate(validation_dataset.as_numpy_iterator()):
                    x_batch = batch[0].copy()
                    batch_size = tf.shape(x_batch)[0]
                    y_batch = batch[1]
                    labels_list.append(y_batch)

                    y_pred = self.predict(x_batch, batch_size=batch_size)
                    labels_clean_list.append(y_pred)
                    loss = self.classifier.loss_object(y_batch, y_pred)
                    normal_loss_sum += loss * tf.cast(batch_size, tf.float32)

                    x_adv = self._attack.generate(x_batch, y_batch)
                    y_adv_pred = self.predict(x_adv, batch_size=batch_size)
                    labels_adv_list.append(y_adv_pred)
                    loss_adv = self.classifier.loss_object(y_batch, y_adv_pred)
                    adv_loss_sum += loss_adv * tf.cast(batch_size, tf.float32)

                    sum_batches += batch_size

                normal_loss_avg = normal_loss_sum / tf.cast(sum_batches, tf.float32)
                adv_loss_avg = adv_loss_sum / tf.cast(sum_batches, tf.float32)
                labels = tf.concat(labels_list, axis=0)
                labels_clean = tf.concat(labels_clean_list, axis=0)
                labels_adv = tf.concat(labels_adv_list, axis=0)
                clean_acc = np.sum(np.argmax(labels_clean, axis=1) == np.argmax(labels, axis=1))/sum_batches
                adv_acc = np.sum(np.argmax(labels_clean, axis=1) == np.argmax(labels_adv, axis=1))/sum_batches

                print("Clean loss: ", normal_loss_avg)
                print("Adv loss:", adv_loss_avg)
                print("Clean acc: ", clean_acc)
                print("Adv acc: ", adv_acc)

                logger.info(
                    "epoch: %s time(s): %.1f loss: %.4f acc-adv (tr): %.4f acc-clean (val): %.4f acc-adv (val): %.4f",
                    i_epoch,
                    train_time - start_time,
                    train_loss / train_n,
                    train_acc / train_n,
                    clean_acc,
                    adv_acc,
                )

                # # save last checkpoint
                # if i_epoch + 1 == nb_epochs:
                #     self._classifier.save(filename=f"awp_{self._mode.lower()}_epoch_{i_epoch}")
                #
                # # save best checkpoint
                # if adv_acc > best_acc_adv_test:
                #     self._classifier.save(filename=f"awp_{self._mode.lower()}_epoch_best")
                #     best_acc_adv_test = adv_acc

            else:
                logger.info(
                    "epoch: %s time(s): %.1f loss: %.4f acc-adv: %.4f",
                    i_epoch,
                    train_time - start_time,
                    train_loss / train_n,
                    train_acc / train_n,
                )
            for callback in callbacks:
                logs = {"loss": train_loss / train_n, "accuracy": train_acc / train_n}
                if validation_dataset is not None:
                    logs["val_loss"] = normal_loss_avg
                    logs["val_accuracy"] = clean_acc
                    logs["adv_val_loss"] = adv_loss_avg
                    logs['adv_val_accuracy'] = adv_acc
                callback.on_epoch_end(i_epoch, logs)

        for callback in callbacks:
            logs = {"loss": train_loss / train_n, "acc": train_acc / train_n}
            callback.on_train_end(logs)


    def fit_generator(
            self,
            generator: DataGenerator,
            validation_data: tuple[np.ndarray, np.ndarray] | None = None,
            nb_epochs: int = 20,
            scheduler: None = None,
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

        logger.info("Performing adversarial training with AWP with %s protocol", self._mode)

        size = generator.size
        batch_size = generator.batch_size
        if size is not None:
            nb_batches = int(np.ceil(size / batch_size))
        else:
            raise ValueError("Size is None.")

        logger.info("Adversarial Training AWP with %s", self._mode)

        best_acc_adv_test = 0
        for i_epoch in trange(nb_epochs, desc=f"Adversarial Training AWP with {self._mode} - Epochs"):

            if i_epoch >= self._warmup:
                self._apply_wp = True

            start_time = time.time()
            train_loss = 0.0
            train_acc = 0.0
            train_n = 0.0

            for _ in range(nb_batches):
                # Create batch data
                x_batch, y_batch = generator.get_batch()
                x_batch = x_batch.copy()

                _train_loss, _train_acc, _train_n = self._batch_process(x_batch, y_batch)

                train_loss += _train_loss
                train_acc += _train_acc
                train_n += _train_n

            train_time = time.time()

            # compute accuracy
            if validation_data is not None:
                (x_test, y_test) = validation_data
                y_test = check_and_transform_label_format(y_test, nb_classes=self.classifier.nb_classes)

                x_preprocessed_test, y_preprocessed_test = self._classifier._apply_preprocessing(
                    x_test,
                    y_test,
                    fit=True,
                )
                # pylint: enable=protected-access
                output_clean = np.argmax(self.predict(x_preprocessed_test), axis=1)
                nb_correct_clean = np.sum(output_clean == np.argmax(y_preprocessed_test, axis=1))
                x_test_adv = self._attack.generate(x_preprocessed_test, y=y_preprocessed_test)
                output_adv = np.argmax(self.predict(x_test_adv), axis=1)
                nb_correct_adv = np.sum(output_adv == np.argmax(y_preprocessed_test, axis=1))

                logger.info(
                    "epoch: %s time(s): %.1f loss: %.4f acc-adv (tr): %.4f acc-clean (val): %.4f acc-adv (val): %.4f",
                    i_epoch,
                    train_time - start_time,
                    train_loss / train_n,
                    train_acc / train_n,
                    nb_correct_clean / x_test.shape[0],
                    nb_correct_adv / x_test.shape[0],
                )
                # save last checkpoint
                if i_epoch + 1 == nb_epochs:
                    self._classifier.save(filename=f"awp_{self._mode.lower()}_epoch_{i_epoch}")

                # save best checkpoint
                if nb_correct_adv / x_test.shape[0] > best_acc_adv_test:
                    self._classifier.save(filename=f"awp_{self._mode.lower()}_epoch_best")
                    best_acc_adv_test = nb_correct_adv / x_test.shape[0]

            else:
                logger.info(
                    "epoch: %s time(s): %.1f loss: %.4f acc-adv: %.4f",
                    i_epoch,
                    train_time - start_time,
                    train_loss / train_n,
                    train_acc / train_n,
                )

    def _validate(
            self,
            i_epoch: int,
            validation_data: list[np.ndarray, np.ndarray] | None = None
    ):
        def validation_decorator(func):
            if validation_data is not None:
                def wrapper(*args, **kwargs):
                    (x_test, y_test) = validation_data
                    y_test = check_and_transform_label_format(y_test, nb_classes=self.classifier.nb_classes)
                    x_preprocessed_test, y_preprocessed_test = self._classifier._apply_preprocessing(x_test, y_test,
                                                                                                     fit=True)
                    train_time, train_loss, train_acc = func(*args, **kwargs)
                    output = np.argmax(self.predict(x_preprocessed_test), axis=1)
                    nb_correct_pred = np.sum(output == np.argmax(y_preprocessed_test, axis=1))
                    logger.info(
                        "epoch: %s time(s): %.1f loss: %.4f acc(tr): %.4f acc(val): %.4f",
                        i_epoch,
                        train_time,
                        train_loss,
                        train_acc,
                        nb_correct_pred / y_preprocessed_test.shape[0],
                    )
            else:
                def wrapper(*args, **kwargs):
                    train_time, train_loss, train_acc = func(*args, **kwargs)
                    logger.info(
                        "epoch: %s time(s): %.1f loss: %.4f acc: %.4f",
                        i_epoch,
                        train_time,
                        train_loss,
                        train_acc
                    )
            return wrapper

        return validation_decorator

    def _epoch_step_for_dataset(self, x: np.ndarray, y: np.ndarray, batch_size: int, nb_batches: int,
                                ind: np.ndarray) -> tuple[float, float, float]:
        """
        Tracks batch measurements. Data is provided by x, y numpy ndarray.

        :param x: Batch of x.
        :param y: Batch of y.
        :param batch_size: Size of the batch.
        :param nb_batches: Number of batches.
        :param ind: Indices of given data. They are shuffled before training on whole batch.
        :return: Tuple containing processing time, average loss and average accuracy for current epoch.
        """
        np.random.shuffle(ind)
        start_time = time.time()
        train_loss = 0.0
        train_acc = 0.0
        train_n = 0.0

        for batch_id in range(nb_batches):
            x_batch = x[ind[batch_id * batch_size: min((batch_id + 1) * batch_size, x.shape[0])]].copy()
            y_batch = y[ind[batch_id * batch_size: min((batch_id + 1) * batch_size, x.shape[0])]]

            _train_loss, _train_acc, _train_n = self._batch_process(x_batch, y_batch)

            train_loss += _train_loss
            train_acc += _train_acc
            train_n += _train_n

        train_time = time.time()
        return train_time - start_time, train_loss / train_n, train_acc / train_n

    def _epoch_step_for_generator(self, generator: DataGenerator, nb_batches: int) -> tuple[float, float, float]:
        """
        Tracks batch measurements.

        :param generator: Generator of type DataGenerator.
        :param nb_batches: Number of batches.
        :return: Tuple containing processing time, average loss and average accuracy for current epoch.
        """
        start_time = time.time()
        train_loss = 0.0
        train_acc = 0.0
        train_n = 0.0

        for batch_id in range(nb_batches):
            x_batch, y_batch = generator.get_batch()
            x_batch = x_batch.copy()

            _train_loss, _train_acc, _train_n = self._batch_process(x_batch, y_batch)

            train_loss += _train_loss
            train_acc += _train_acc
            train_n += _train_n

        train_time = time.time()
        return train_time - start_time, train_loss / train_n, train_acc / train_n

    def _batch_process(self, x_batch: np.ndarray, y_batch: np.ndarray, adversarial_perturbed_weights: AWPProtocolTF) -> \
            tuple[float, float, float]:

        train_loss, train_acc, train_n = adversarial_perturbed_weights.awp_step(x_batch, y_batch)
        # train_acc = keras.metrics.categorical_accuracy(o_batch, model_outputs_pert)
        return tf.get_static_value(train_loss), tf.get_static_value(train_acc), tf.get_static_value(train_n)
