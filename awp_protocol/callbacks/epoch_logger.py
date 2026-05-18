import os
import datetime

import tensorflow as tf

from awp_protocol import awp

def _datetime_now() -> str:
    d = datetime.datetime.now()
    return d.strftime("%Y-%m-%d %H:%M:%S")


class EpochLogger(tf.keras.callbacks.Callback):

    def __init__(self, save_filepath, attack_params=None, training_params=None):
        super().__init__()
        self._save_filepath = save_filepath
        self._attack_params = attack_params
        self._training_params: awp.Params | None = training_params
        self._init_log_file()


    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self._register_epoch(epoch, logs)


    def _init_log_file(self):
        os.makedirs(os.path.dirname(self._save_filepath), exist_ok=True)
        with open(self._save_filepath, "w") as log_file:
            log_file.write(f"attack parameters: {self._attack_params}, training parameters: {self._training_params}")
            log_file.write("Epoch\tTrain Time\tTest Time\tLR\tTrain Loss\tTrain Acc\tTrain Robust Loss\tTrain Robust Acc\tTest Loss\tTest Acc\tTest Robust Loss\tTest Robust Acc\n")


    def _register_epoch(self, epoch: int, logs: dict):
        datetime_prefix = _datetime_now()
        with open(self._save_filepath, "a") as log_file:
            log_file.write(f"{datetime_prefix} - {epoch}:\t{logs.get("train_time")}\t{logs.get("val_time")}\t{"lr"}\t"
                           f"{logs.get("loss")}\t{logs.get("accuracy")}\t{logs.get("robust_loss")}\t{logs.get("robust_accuracy")}\t"
                           f"{logs.get("val_loss")}\t{logs.get("accuracy")}\t{logs.get("val_robust_loss")}\t{logs.get("val_robust_accuracy")}\n")
