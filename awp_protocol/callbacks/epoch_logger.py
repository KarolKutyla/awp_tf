import os
import tensorflow as tf


class ProgBarLogger(tf.keras.callbacks.Callback):

    def __init__(self, save_filepath, batch_interval=20):
        super().__init__()
        self._save_filepath = save_filepath
        os.makedirs(self._save_filepath, exist_ok=True)


    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        values = self._collect_train_metrics(logs)
        values.append(self._collect_validation_metrics(logs))



    def _collect_train_metrics(self, logs: dict):
        values = []
        if "loss" in logs:
            values.append(("loss", float(logs["loss"])))
        if "accuracy" in logs:
            values.append(("acc", float(logs["accuracy"])))
        if "robust_loss" in logs:
            values.append(("robust_loss", float(logs["robust_loss"])))
        if "robust_accuracy" in logs:
            values.append(("robust_acc", float(logs["robust_accuracy"])))
        return values


    def _collect_validation_metrics(self, logs: dict):
        values = []
        if "val_loss" in logs:
            values.append(("loss", float(logs["val_loss"])))
        if "val_accuracy" in logs:
            values.append(("val_accuracy", float(logs["val_accuracy"])))
        if "val_robust_loss" in logs:
            values.append(("val_robust_loss", float(logs["val_robust_loss"])))
        if "val_robust_acc" in logs:
            values.append(("val_robust_acc", float(logs["val_robust_acc"])))