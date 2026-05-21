import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# ---------------------------
# Residual Block
# ---------------------------
class WRNBlock(layers.Layer):
    def __init__(self, filters, stride, drop_rate=0.0):
        super().__init__()
        self.filters = filters
        self.stride = stride
        self.drop_rate = drop_rate

        self.bn1 = layers.BatchNormalization()
        self.conv1 = layers.Conv2D(filters, 3, stride, padding="same", use_bias=False)

        self.bn2 = layers.BatchNormalization()
        self.conv2 = layers.Conv2D(filters, 3, 1, padding="same", use_bias=False)

        self.dropout = layers.Dropout(drop_rate) if drop_rate > 0 else None

        self.shortcut = None

    def build(self, input_shape):
        in_channels = input_shape[-1]
        if in_channels != self.filters or self.stride != 1:
            self.shortcut = layers.Conv2D(
                self.filters, 1, self.stride, padding="same", use_bias=False
            )

    def call(self, x, training=False):
        shortcut = x

        # Pre-activation style (WRN)
        out = self.bn1(x, training=training)
        out = tf.nn.relu(out)
        out = self.conv1(out)

        out = self.bn2(out, training=training)
        out = tf.nn.relu(out)

        if self.dropout:
            out = self.dropout(out, training=training)

        out = self.conv2(out)

        if self.shortcut is not None:
            shortcut = self.shortcut(shortcut)

        return out + shortcut


# ---------------------------
# WRN Block group
# ---------------------------
def make_block_group(x, filters, num_blocks, stride, drop_rate):
    x = WRNBlock(filters, stride, drop_rate)(x)
    for _ in range(num_blocks - 1):
        x = WRNBlock(filters, 1, drop_rate)(x)
    return x


# ---------------------------
# WRN-28-10 model
# ---------------------------
def WideResNet28_10(input_shape=(32, 32, 3), num_classes=10, drop_rate=0.0):
    inputs = keras.Input(shape=input_shape)

    # stem
    x = layers.Conv2D(16, 3, padding="same", use_bias=False)(inputs)

    # WRN-28 = (4,4,4) blocks
    # widen_factor=10 => 160/320/640 channels
    x = make_block_group(x, 160, num_blocks=4, stride=1, drop_rate=drop_rate)
    x = make_block_group(x, 320, num_blocks=4, stride=2, drop_rate=drop_rate)
    x = make_block_group(x, 640, num_blocks=4, stride=2, drop_rate=drop_rate)

    x = layers.BatchNormalization()(x)
    x = tf.nn.relu(x)

    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(num_classes)(x)

    return keras.Model(inputs, outputs, name="WRN_28_10")