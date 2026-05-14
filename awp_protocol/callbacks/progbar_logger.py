import tensorflow as tf


class ProgbarLogger(tf.keras.callbacks.Callback):

    def __init__(self, batch_interval=20):
        super().__init__()
        self._progbar: tf.keras.utils.Progbar | None = None
        self.batch_interval = batch_interval


    def update_progbar(self, progbar: tf.keras.utils.Progbar):
        self._progbar = progbar


    def on_batch_end(self, batch, logs=None):
        logs = logs or {}

        if batch % self.batch_interval == 0 and self._progbar is not None:
            values = self._collect_train_metrics(logs)
            self._progbar.update(batch, values)


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


