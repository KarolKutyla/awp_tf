from awp_protocol.attacks.attack import TensorflowEvasionAttack
import numpy as np
from matplotlib import pyplot as plt

import tensorflow as tf

class AdversarialPlots:

    def __init__(self, attack: TensorflowEvasionAttack, labels: dict[int, str] | None = None):
        self._attack = attack
        self._labels = labels

    def generate_and_show_adversarial_batch(self, x: tf.Tensor, y: tf.Tensor) -> None:
        x_adv = self._attack.generate(x, y)
        self.show_adversarial_batch(x, x_adv)

    def show_adversarial_batch(self, x: tf.Tensor, x_adv: tf.Tensor, n=8) -> None:
        x = x[:n]
        x_adv=x_adv[:n]
        indices, probs = self._probs_and_indices(x)
        indices_adv, probs_adv = self._probs_and_indices(x_adv)

        x: np.ndarray = x.numpy()
        adv = x_adv.numpy()
        x = (x * 255).astype(np.uint8)
        adv = (adv * 255).astype(np.uint8)

        for i in range(n):
            plt.figure(figsize=(12, 4))
            plt.axis("off")

            ax = plt.subplot(1, 2, 1)
            label = indices[i].numpy()
            if self._labels is not None:
                label = self._labels[label]
            ax.set_title(f"Original\nlabel: {label}\nwith probability {probs[i].numpy():.3f}")
            plt.imshow(x[i])

            ax = plt.subplot(1, 2, 2)
            label = indices_adv[i].numpy()
            if self._labels is not None:
                label = self._labels[label]
            ax.set_title(f"Adversarial\nlabel: {label}\nwith probability {probs_adv[i].numpy():.3f}")
            plt.imshow(adv[i])

            plt.tight_layout()
            plt.show()


    def _probs_and_indices(self, x: tf.Tensor):
        logits = self._attack.model(x)
        probs = tf.nn.softmax(logits)
        max_probs = tf.reduce_max(probs, axis=-1)
        indices = tf.argmax(probs, axis=-1)
        rounded_probs = tf.round(max_probs * 1000) / 1000
        return indices, rounded_probs