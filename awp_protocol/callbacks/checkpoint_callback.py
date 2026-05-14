import os
import tensorflow as tf


class EpochCheckpoint(tf.keras.callbacks.Callback):

    def __init__(self, save_dir, verbose=False):
        super().__init__()
        self.save_dir = save_dir
        self._verbose = verbose
        os.makedirs(save_dir, exist_ok=True)


    def on_epoch_end(self, epoch, logs=None):
        path = os.path.join(self.save_dir, f"epoch_{epoch}")
        self.model.save(path, include_optimizer=True)

        if self._verbose:
            print(f"\n[Checkpoint] saved to {path}")