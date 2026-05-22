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
        .batch(64, drop_remainder=False)
        .prefetch(tf.data.AUTOTUNE)
    )
    test_ds = (
        ds_test
        .map(preprocess_test, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(64, drop_remainder=False)
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

def load_cifar_dataset():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    tf_train_ds = (
        tf.data.Dataset.from_tensor_slices((x_train, y_train))
        .shuffle(50000)
        .map(transform_train_from_research_paper, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(128, drop_remainder=False)
        .prefetch(tf.data.AUTOTUNE)
    )
    tf_test_ds = (
        tf.data.Dataset.from_tensor_slices((x_test, y_test))
        .map(lambda x, y: (tf.keras.applications.resnet_v2.preprocess_input(tf.cast(x, tf.float32)), tf.cast(y, tf.float32)), num_parallel_calls=tf.data.AUTOTUNE)
        .batch(128, drop_remainder=False)
        .prefetch(tf.data.AUTOTUNE)
    )

    return tf_train_ds, tf_test_ds


def transform_train_from_research_paper(image, label):
    image = tf.cast(image, tf.float32)
    image = tf.keras.applications.resnet_v2.preprocess_input(image)
    image = tf.pad(
        image,
        paddings=[[4, 4], [4, 4], [0, 0]],
        mode="CONSTANT",
        constant_values=0
    )

    image = tf.image.random_crop(
        image,
        size=[32, 32, 3]
    )

    image = tf.image.random_flip_left_right(image)

    return image, tf.cast(label, image.dtype)
