import os
import tensorflow as tf


class EpochCheckpoint(tf.keras.callbacks.Callback):

    def __init__(self, save_dir, interval = 10, verbose=False):
        super().__init__()
        self._save_dir = save_dir
        self._verbose = verbose
        self._interval = interval
        os.makedirs(self._save_dir, exist_ok=True)


    def on_epoch_end(self, epoch, logs=None):
        if epoch % self._interval == 0:
            path = os.path.join(self._save_dir, f"epoch_{epoch}.keras")
            self.model.save(path, include_optimizer=True)

            if self._verbose:
                print(f"\n[Checkpoint] saved to {path}")