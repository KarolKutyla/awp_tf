from random import randrange

import numpy as np

import tensorflow as tf
from tensorflow import keras

import tensorflow_datasets as tfds
from tensorflow.keras.datasets import mnist

import torch

def load_cifar_dataset():
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
    x_train = x_train / 255.0
    x_test = x_test / 255.0

    seed = randrange(1, 1000000)
    np.random.seed(seed)
    torch.manual_seed(seed)
    tf.random.set_seed(seed)

    tf_x_train = x_train.astype(np.float32)
    tf_x_test = x_test.astype(np.float32)

    tf_train_ds = tf.data.Dataset.from_tensor_slices((tf_x_train, y_train))
    tf_train_ds = tf_train_ds.shuffle(10000).batch(64).prefetch(tf.data.AUTOTUNE)
    tf_train_ds = tf_train_ds.cache().prefetch(tf.data.AUTOTUNE)

    torch_train_loader = _convert_to_torch_loader(x_train, y_train, shuffle=True)
    torch_test_loader = _convert_to_torch_loader(x_test, y_test)
    return tf_train_ds, x_test, y_test, torch_train_loader, torch_test_loader

def load_mnist_dataset():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train / 255.0
    x_test = x_test / 255.0
    # y_train = y_train.reshape(-1)
    # y_test = y_test.reshape(-1)

    seed = randrange(1, 1000000)
    np.random.seed(seed)
    torch.manual_seed(seed)
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
    tf_train_ds = tf_train_ds.shuffle(10000).batch(64).prefetch(tf.data.AUTOTUNE)
    tf_train_ds = tf_train_ds.cache().prefetch(tf.data.AUTOTUNE)

    torch_train_loader = _convert_to_torch_loader(x_train, y_train, shuffle=True)
    torch_test_loader = _convert_to_torch_loader(x_test, y_test)
    return tf_train_ds, x_test, y_test, torch_train_loader, torch_test_loader

def _convert_to_torch_loader(x, y, shuffle=False):
    x_torch = torch.from_numpy(x).permute(0, 3, 1, 2).float()
    y_torch = torch.from_numpy(y).squeeze().long()
    torch_test_dataset = torch.utils.data.TensorDataset(x_torch, y_torch)
    torch_loader = torch.utils.data.DataLoader(
        torch_test_dataset,
        batch_size=64,
        shuffle=shuffle
    )
    return torch_loader
