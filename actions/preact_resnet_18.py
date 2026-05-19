import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# ---------------------------
# PreAct Residual Block
# ---------------------------
def preact_resnet_block(x, filters, stride=1):
    bn1 = layers.BatchNormalization()(x)
    act1 = layers.Activation("relu")(bn1)

    shortcut = x

    # first conv
    y = layers.Conv2D(filters, 3, stride, padding="same", use_bias=False)(act1)
    y = layers.BatchNormalization()(y)
    y = layers.Activation("relu")(y)

    # second conv
    y = layers.Conv2D(filters, 3, 1, padding="same", use_bias=False)(y)

    # shortcut projection if needed
    if stride != 1 or x.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, 1, stride, use_bias=False)(act1)

    out = layers.Add()([y, shortcut])
    return out


# ---------------------------
# Stage (stack of blocks)
# ---------------------------
def make_stage(x, filters, blocks, stride):
    x = preact_resnet_block(x, filters, stride=stride)
    for _ in range(1, blocks):
        x = preact_resnet_block(x, filters, stride=1)
    return x


# ---------------------------
# PreActResNet18
# ---------------------------
def PreActResNet18(input_shape=(32, 32, 3), num_classes=10, width_mult=1):
    inputs = keras.Input(shape=input_shape)

    # initial conv
    x = layers.Conv2D(64, 3, padding="same", use_bias=False)(inputs)

    # ResNet18 structure
    x = make_stage(x, 64 * width_mult, 2, stride=1)
    x = make_stage(x, 128 * width_mult, 2, stride=2)
    x = make_stage(x, 256 * width_mult, 2, stride=2)
    x = make_stage(x, 512 * width_mult, 2, stride=2)

    # head
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.GlobalAveragePooling2D()(x)

    outputs = layers.Dense(num_classes)(x)

    return keras.Model(inputs, outputs, name=f"PreActResNet18_w{width_mult}")