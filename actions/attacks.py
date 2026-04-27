from awp_protocol.attacks.pgd import PGDAttack
import numpy as np
from matplotlib import pyplot as plt

import tensorflow as tf

def show_adversarial_batch(x: tf.Tensor, x_adv: tf.Tensor, n=8):
    x: np.ndarray = x.numpy()
    adv = x_adv.numpy()

    # denormalization [0,1] -> [0,255]
    x = (x * 255).astype(np.uint8)
    adv = (adv * 255).astype(np.uint8)

    plt.figure(figsize=(12, 4))

    for i in range(n):
        # original
        plt.subplot(2, n, i + 1)
        plt.imshow(x[i])
        plt.axis("off")
        if i == 0:
            plt.title("Original")

        # adversarial
        plt.subplot(2, n, i + 1 + n)
        plt.imshow(adv[i])
        plt.axis("off")
        if i == 0:
            plt.title("PGD")

    plt.tight_layout()
    plt.show()