from __future__ import absolute_import, division, print_function, unicode_literals, annotations

from abc import ABC, abstractmethod
import time

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import Callback

from awp_tf.attacks.attack import EvasionAttack
from awp_tf.callbacks.progbar_logger import ProgbarLogger
from awp_tf.callbacks.checkpoint_callback import EpochCheckpoint
from awp_tf.losses.loss import AdversarialLoss


class Trainer(ABC):

    def __init__(
            self,
            classifier: tf.keras.Model,
            attack: EvasionAttack,
            adversarial_loss: AdversarialLoss
    ):
        self._fast_mode = True

        self._classifier: tf.keras.Model = classifier
        self._attack: EvasionAttack = attack
        self._robust_loss: AdversarialLoss = adversarial_loss

        self._steps_per_epoch: int | None = None
        self._epochs_run = 0

        self._progbar: tf.keras.utils.Progbar
        self._callback_list: tf.keras.callbacks.CallbackList
        self._logger: ProgbarLogger
        self._ckpt = EpochCheckpoint(self._classifier.name)

        self._clean_loss_metric = tf.keras.metrics.Mean()
        self._clean_accuracy_metric = tf.keras.metrics.SparseCategoricalAccuracy()
        self._robust_loss_metric = tf.keras.metrics.Mean()
        self._robust_accuracy_metric = tf.keras.metrics.SparseCategoricalAccuracy()

        _validate_optimizer(classifier)


    def fit(
            self,
            x: tf.Tensor,
            y: tf.Tensor,
            validation_data: tuple[tf.Tensor, tf.Tensor] | None = None,
            batch_size: int = 128,
            nb_epochs: int = 1,
            callbacks: list[Callback] | None = None,
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
            callbacks: list[tf.keras.callbacks.Callback] | None = None,
    ):
        self._steps_per_epoch = train_dataset.cardinality().numpy() or None
        self._train_loop(train_dataset, nb_epochs, callbacks=callbacks, validation_dataset=validation_dataset)


    def _train_loop(
            self,
            train_dataset,
            nb_epochs,
            validation_dataset: tf.data.Dataset | None = None,
            callbacks: list[tf.keras.callbacks.Callback] | None = None,
    ):
        callbacks = callbacks or []
        self._logger = ProgbarLogger()
        callbacks += [self._logger]
        self._callback_list = tf.keras.callbacks.CallbackList(callbacks, add_history=True, model=self._classifier)

        self._callback_list.on_train_begin()

        for epoch in range(nb_epochs):
            self._epoch(train_dataset, epoch + 1, validation_dataset=validation_dataset)

        self._callback_list.on_train_end()


    def _epoch(self, train_dataset: tf.data.Dataset, epoch: int, validation_dataset: tf.data.Dataset | None = None, ):
        self._reset_metrics()

        self._progbar = tf.keras.utils.Progbar(
            self._steps_per_epoch,
            stateful_metrics=["loss", "accuracy", "robust_loss", "robust_accuracy"],
        )
        self._logger.update_progbar(self._progbar)

        self._callback_list.on_epoch_begin(self._epochs_run)

        start_time = time.time()
        self._train_batches(train_dataset)
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

    @abstractmethod
    def _train_batches(self, dataset: tf.data.Dataset):
        ...

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
            batch_results = self._validation(x_batch, y_batch)
            self._update_metrics(y_batch, batch_results)

    def _validation(self, x_batch, y_batch):
        x_batch_adv = self._attack.generate(x_batch, y_batch)
        logits_clean = self._classifier(x_batch, training=False)
        loss_on_clean_examples = self._classifier.loss(y_true=y_batch, y_pred=logits_clean)
        logits_adv = self._classifier(x_batch_adv, training=False)
        loss_on_adversarial_examples = self._classifier.loss(y_true=y_batch, y_pred=logits_adv)

        return loss_on_clean_examples, logits_clean, loss_on_adversarial_examples, logits_adv


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


def _validate_optimizer(classifier: keras.models.Model):
    if classifier.optimizer is None:
        raise Exception(
            "No optimizer provided for the classifier. For native awp compile your model with SGD with custom learning rate and 0.0 momentum.")

    if not classifier.optimizer.built:
        classifier.optimizer.build(classifier.trainable_variables)