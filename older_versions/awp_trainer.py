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
import tensorflow as tf
from tensorflow import keras
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
            **kwargs,
    ):
        """
        Train a model adversarially with AWP protocol.
        See class documentation for more information on the exact procedure.

        :param x: Training set.
        :param y: Labels for the training set.
        :param validation_data: Tuple consisting of validation data, (x_val, y_val)
        :param batch_size: Size of batches.
        :param nb_epochs: Number of epochs to use for trainings.
        :param kwargs: Dictionary of framework-specific arguments. These will be passed as such to the `fit` function of
                                  the target classifier.
        """

        logger.info("Performing adversarial training with AWP with %s protocol", self._mode)

        best_acc_adv_test = 0
        nb_batches = int(np.ceil(len(x) / batch_size))
        ind = np.arange(len(x))

        logger.info("Adversarial Training AWP with %s", self._mode)
        y = check_and_transform_label_format(y, nb_classes=self.classifier.nb_classes)
        # adversarial_perturbed_weights = self.AWPProtocolTF(self._classifier, self._classifier.loss_object,
        #                                                    attack=self._attack, optimizer=self._classifier.optimizer)
        adversarial_perturbed_weights = self.AWPProtocolTF(self._classifier.model, self._classifier.loss_object, optimizer=self._classifier.optimizer, attack=self._attack)

        for i_epoch in trange(nb_epochs, desc=f"Adversarial Training AWP with {self._mode} - Epochs"):

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
                x_batch = x[ind[batch_id * batch_size: min((batch_id + 1) * batch_size, x.shape[0])]].copy()
                y_batch = y[ind[batch_id * batch_size: min((batch_id + 1) * batch_size, x.shape[0])]]

                _train_loss, _train_acc, _train_n = self._batch_process(x_batch, y_batch, adversarial_perturbed_weights)

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

    def fit_generator(
            self,
            generator: DataGenerator,
            validation_data: tuple[np.ndarray, np.ndarray] | None = None,
            nb_epochs: int = 20,
            scheduler: None = None,
            **kwargs,
    ):
        """
        Train a model adversarially with AWP protocol using a data generator.
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

    def _weight_perturbation(
            self, x_batch: "tf.Tensor", x_batch_pert: "tf.Tensor", y_batch: "tf.Tensor"
    ) -> dict[str, "tf.Tensor"]:
        """
        Calculate wight perturbation for a batch of data.
        See class documentation for more information on the exact procedure.

        :param x_batch: batch of x.
        :param x_batch_pert: batch of x with perturbations.
        :param y_batch: batch of y.
        :return: dict containing names of classifier model's layers as keys and parameters as values
        """

        w_perturb = OrderedDict()
        params_dict, _ = self._calculate_model_params(self._classifier)
        list_keys = list(params_dict.keys())
        self._proxy_classifier.model.set_weights(self._classifier.model.trainable_weights)

        if self._mode.lower() == "pgd":
            # Perform prediction
            with tf.GradientTape() as tape:
                model_outputs_pert = self._proxy_classifier.model(x_batch_pert)
                loss = self._proxy_classifier.loss_object(y_batch, model_outputs_pert)
            grad = tape.gradient(loss, self._proxy_classifier.model.trainable_weights)
            tape.stop_recording()
        elif self._mode.lower() == "trades":
            with tf.GradientTape() as tape:
                model_outputs = self._proxy_classifier.model(x_batch, training=True)
                model_outputs_pert = self._proxy_classifier.model(x_batch_pert, training=True)
                loss_clean = self._proxy_classifier.loss_object(y_batch, model_outputs)
                loss_kl = keras.losses.KLDivergence(reduction="sum_over_batch_size")(
                    tf.clip_by_value(tf.nn.softmax(model_outputs, axis=1), clip_value_min=EPS,
                                     clip_value_max=tf.dtypes.as_dtype(model_outputs.dtype).max),
                    tf.nn.softmax(model_outputs_pert, axis=1)
                )
                loss = loss_clean + self._beta * loss_kl
                # loss = -1.0 * (loss_clean + self._beta * loss_kl)
            grad = tape.gradient(loss, self.classifier.model.trainable_weights)
            tape.stop_recording()

        else:
            raise ValueError(
                "Incorrect mode provided for base adversarial training. 'mode' must be among 'PGD' and 'TRADES'."
            )

        self._proxy_classifier.optimizer.apply(grad, self._proxy_classifier.model.trainable_weights)

        params_dict_proxy, _ = self._calculate_model_params(self._proxy_classifier)

        for name in list_keys:
            perturbation = params_dict_proxy[name]["param"] - params_dict[name]["param"]
            print(list(params_dict[name]["size"]))
            perturbation = tf.reshape(perturbation, list(params_dict[name]["size"]))
            scale = params_dict[name]["norm"].numpy()[0] / (tf.norm(perturbation) + EPS)
            w_perturb[name] = scale * perturbation

        return w_perturb

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

    class AWPProtocolTF:
        EPS = 1e-8
        from tensorflow.python.eager.def_function import Function as TfFunction
        PARAMS_DICT = {'weight_constraint': 0.01, 'awp_step_size': None, 'awp_steps': 1, 'alternate_iteration': 1,
                       'learning_rate': 0.01, 'pgd_step': 10, 'perturbation_bound': 8/255, "pgd_step_size": 0.1, "mode": "trades"}

        def __init__(self, classifier: keras.Model, loss: TfFunction, attack=None, optimizer=None,
                     tracked_layers: list[bool] | None = None, params_dict: dict = PARAMS_DICT):
            self._classifier = classifier
            self._proxy_classifier = tf.keras.models.clone_model(classifier)
            self._proxy_classifier.set_weights(classifier.get_weights())
            self._tracked_layers = [variable.name == 'kernel' for variable in
                                    classifier.trainable_variables] if tracked_layers is None else tracked_layers
            self._loss_obj = loss if loss is not None else tf.keras.losses.CategoricalCrossentropy()
            if params_dict["mode"] == "pgd":
              self._loss = self._loss_pgd
            elif params_dict["mode"] == "trades":
              self._loss = self._loss_trades
            if self._loss is None:
              raise Exception("Mode not provided! Chose pgd or trades.")
            self._trades_beta = 0.1

            self._attack = attack
            if self._attack is None:
              self._attack = self._attack_func
              self.awp_step = self._awp_step_tf
            else:
              self.awp_step = self._awp_step_non_tf
            self._learning_rate = params_dict["learning_rate"]
            self._optimizer = self._clone_init_optimizer(optimizer) if optimizer is not None else None
            self._weight_constraint = params_dict["weight_constraint"]
            self._alternate_iteration = params_dict["alternate_iteration"]
            self._awp_steps = params_dict["awp_steps"]
            self._pgd_step = params_dict["pgd_step"]
            self._pgd_step_size = params_dict["pgd_step_size"]
            self._perturbation_bound = params_dict['perturbation_bound']
            self._awp_step_size = params_dict["awp_step_size"] if params_dict[
                                                                      "awp_step_size"] is not None else self._weight_constraint / (
                    self._awp_steps * self._alternate_iteration)

            self._weight_norms = [tf.Variable(tf.norm(variables)) if tracked else None for variables, tracked in
                                  zip(self._classifier.trainable_variables, self._tracked_layers)]
            self._weight_perturbation_sizes = [tf.Variable(weight_size * self._weight_constraint) if tracked else None
                                               for weight_size, tracked in
                                               zip(self._weight_norms, self._tracked_layers)]

        def _clone_init_optimizer(self, optimizer):
          cfg = optimizer.get_config()
          opt = optimizer.__class__.from_config(cfg)
          zero_grads = [tf.zeros_like(v) for v in self._proxy_classifier.trainable_variables]
          opt.learning_rate.assign(self._learning_rate)
          opt.apply_gradients(zip(zero_grads, self._proxy_classifier.trainable_variables))
          return opt

        @tf.function
        def _loss_pgd(self, x: tf.Tensor, x_pert: tf.Tensor, y: tf.Tensor):
          y_pred = self._proxy_classifier(x_pert, training=True)
          return self._loss_obj(y, y_pred), y_pred

        @tf.function
        def _loss_trades(self, x: tf.Tensor, x_pert: tf.Tensor, y: tf.Tensor):
          model_outputs = self._proxy_classifier(x, training=True)
          model_outputs_pert = self._proxy_classifier(x_pert, training=True)
          loss_clean = self._loss_obj(y, model_outputs)
          loss_kl = keras.losses.KLDivergence(reduction="sum_over_batch_size")(
              tf.clip_by_value(tf.nn.softmax(model_outputs, axis=1), clip_value_min=EPS,
                               clip_value_max=tf.dtypes.as_dtype(model_outputs.dtype).max),
              tf.nn.softmax(model_outputs_pert, axis=1)
              )
          loss = loss_clean + self._trades_beta * loss_kl
          return loss, model_outputs_pert

        def awp_step(self, x: tf.Tensor, y: tf.Tensor):
          ...

        def _awp_step_non_tf(self, x: tf.Tensor, y: tf.Tensor):
            self._reset_state()
            for i in range(self._alternate_iteration):
                if self._attack is not None:
                  x_pert = self._attack.generate(x, y)
                else:
                  x_pert = self._attack_func(x, y)
                for j in range(self._awp_steps):
                    self._find_v(x, x_pert, y)
                    self._trim_v()
            return self._apply_v_to_model(x, x_pert, y)

        @tf.function
        def _awp_step_tf(self, x: tf.Tensor, y: tf.Tensor):
          self._reset_state()
          for i in range(self._alternate_iteration):
            x_pert = self._attack_func(x, y)
            for j in range(self._awp_steps):
              self._find_v(x, x_pert, y)
              self._trim_v()
          return self._apply_v_to_model(x, x_pert, y)

        @tf.function
        def _attack_func(self, x, y):
            def trim_pert_to_bound(t):
                norm = tf.norm(t)
                return tf.cond(norm > self._perturbation_bound, lambda: t * self._perturbation_bound/norm, lambda: t)
            pert = tf.random.uniform(shape=x.shape, minval=-1.0, maxval=1.0, dtype=tf.float32) * self._perturbation_bound
            x_pert = pert + x
            for i in range(self._pgd_step):
                with tf.GradientTape() as tape:
                    tape.watch(x_pert)
                    # y_pred = self._proxy_classifier(x_pert, training=True)
                    loss, _ = self._loss(x, x_pert, y)
                gradient = tape.gradient(loss, x_pert)
                pert = x_pert + gradient * self._pgd_step_size - x
                pert = tf.map_fn(fn=trim_pert_to_bound,  elems=pert)
                x_pert = x + pert

            return x_pert

        @tf.function
        def _reset_state(self):
            for i, tracked in enumerate(self._tracked_layers):
                self._proxy_classifier.trainable_variables[i].assign(
                    self._classifier.trainable_variables[i])
            for i, tracked in enumerate(self._tracked_layers):
                if tracked:
                    self._weight_norms[i].assign(tf.norm(self._classifier.trainable_variables[i]))
                    self._weight_perturbation_sizes[i].assign(self._weight_norms[i] * self._weight_constraint)

        @tf.function
        def _find_v(self, x, x_pert, y_true):
            with tf.GradientTape() as tape:
                # y_pred = self._proxy_classifier(x_pert, training=True)
                loss, _ = self._loss(x, x_pert, y_true)
            gradient = tape.gradient(loss, self._proxy_classifier.trainable_variables)
            for i, tracked in enumerate(self._tracked_layers):
                if tracked:
                    gradient_norm = tf.norm(gradient[i])
                    gradient_norm = tf.maximum(gradient_norm, tf.constant(1e-6, dtype=gradient_norm.dtype))
                    self._proxy_classifier.trainable_variables[i].assign_add(
                        self._awp_step_size * gradient[i] * self._weight_norms[i] / gradient_norm)

        @tf.function
        def _trim_v(self):
            for i, tracked in enumerate(self._tracked_layers):
                v = self._proxy_classifier.trainable_variables[i] - self._classifier.trainable_variables[i]
                if tracked:
                    v_norm = tf.norm(v)
                    pred = v_norm > self._weight_perturbation_sizes[i]

                    def true_fn():
                        self._proxy_classifier.trainable_variables[i].assign(
                            self._classifier.trainable_variables[i] + v * self._weight_perturbation_sizes[i]
                            / tf.maximum(v_norm, tf.constant(1e-6, dtype=v_norm.dtype)))
                        return None
                    tf.cond(pred, true_fn, lambda: None)

        @tf.function
        def _apply_v_to_model(self, x, x_pert, y_true):
            with tf.GradientTape() as tape:
                # y_pred = self._proxy_classifier(x_pert, training=True)
                loss, y_pred = self._loss(x, x_pert, y_true)
            gradient = tape.gradient(loss, self._proxy_classifier.trainable_variables)
            for i, tracked in enumerate(self._tracked_layers):
                if tracked:
                    # g = tf.where(tf.equal(gradient[i], 0.0), tf.constant(1e-6, dtype=gradient[i].dtype), gradient[i])
                    if False:
                      ...
                    # self._optimizer is not None:
                    #     self._optimizer.apply_gradients(
                    #         [(gradient[i], self._proxy_classifier.trainable_variables[i])])
                    #     self._classifier.trainable_variables[i].assign(
                    #         self._proxy_classifier.trainable_variables[i])
                    else:
                        self._classifier.trainable_variables[i].assign_sub(gradient[i] * self._learning_rate)
            train_n = tf.cast(y_true.shape[0], dtype=tf.float32)
            accuracy = tf.reduce_sum(
                tf.cast(tf.argmax(y_pred, axis=1) == tf.argmax(y_true, axis=1), dtype=tf.float32)) / train_n
            return loss, accuracy, train_n
