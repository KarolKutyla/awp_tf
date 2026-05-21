from random import randrange

import numpy as np

import tensorflow as tf
import tensorflow_datasets as tfds

from tensorflow.keras.datasets import mnist


def load_imagenette_dataset():
    (ds_train, ds_test), ds_info = tfds.load(
        "imagenette/320px",  # or "imagenette"
        split=["train", "validation"],
        as_supervised=True,
        with_info=True
    )
    train_ds = (
        ds_train
        .shuffle(10_000)
        .map(preprocess_train, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(128, drop_remainder=False)
        .prefetch(tf.data.AUTOTUNE)
    )
    test_ds = (
        ds_test
        .map(preprocess_test, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(128, drop_remainder=False)
        .prefetch(tf.data.AUTOTUNE)
    )

    return train_ds, test_ds


def preprocess_train(image, label):
    IMG_SIZE = 224

    image = tf.cast(image, tf.float32)

    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))

    image = image / 127.5 - 1.0

    image = tf.image.random_flip_left_right(image)

    image = tf.image.random_brightness(image, 0.2)
    image = tf.image.random_contrast(image, 0.8, 1.2)

    return image, label

def preprocess_test(image, label):
    IMG_SIZE = 224

    image = tf.cast(image, tf.float32)
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    image = image / 127.5 - 1.0
    return image, label

    return x, y

def load_mnist_dataset():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train / 255.0
    x_test = x_test / 255.0
    # y_train = y_train.reshape(-1)
    # y_test = y_test.reshape(-1)

    seed = randrange(1, 1000000)
    np.random.seed(seed)
    tf.random.set_seed(seed)

    x_train = x_train[..., None]  # (N, 28, 28, 1)
    x_train = np.repeat(x_train, 3, axis=-1)  # (N, 28, 28, 3)
    x_train = tf.image.resize(x_train, (32, 32)).numpy()
    x_test = x_test[..., None]  # (N, 28, 28, 1)
    x_test = np.repeat(x_test, 3, axis=-1)  # (N, 28, 28, 3)
    x_test = tf.image.resize(x_test, (32, 32)).numpy()

    tf_x_train = x_train.astype(np.float32)
    tf_x_test = x_test.astype(np.float32)

    tf_train_ds = tf.data.Dataset.from_tensor_slices((tf_x_train, y_train))
    tf_train_ds = tf_train_ds.shuffle(10000).batch(64, drop_remainder=True).prefetch(tf.data.AUTOTUNE)
    tf_train_ds = tf_train_ds.cache().prefetch(tf.data.AUTOTUNE)

    return tf_train_ds, x_test, y_test

def load_cifar_labels():
    return {
        0: "airplane",
        1: "automobile",
        2: "bird",
        3: "cat",
        4: "deer",
        5: "dog",
        6: "frog",
        7: "horse",
        8: "ship",
        9: "truck"
    }