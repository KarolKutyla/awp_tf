import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.initializers import VarianceScaling
import numpy as np

RESIZE_TO = 32


WEIGHT_DECAY = 0.0005
INIT = "he_normal"
DEPTH = 28
WIDTH_MULT = 10

def wide_basic(x, n_input_plane, n_output_plane, stride):

    # Shortcut connection: identity function or 1x1
    # convolutional
    #  (depends on difference between input & output shape - this
    #   corresponds to whether we are using the first block in
    #   each
    #   group; see `block_series()`).

    if n_input_plane != n_output_plane:

        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)

        shortcut = layers.Conv2D(
            n_output_plane,
            (1, 1),
            strides=stride,
            padding="same",
            use_bias=False,
            kernel_initializer=INIT
        )(x)

        convs = layers.Conv2D(
            n_output_plane,
            (3, 3),
            strides=stride,
            padding="same",
            use_bias=False,
            kernel_initializer=INIT
        )(x)

    else:

        shortcut = x

        convs = layers.BatchNormalization()(x)
        convs = layers.Activation("relu")(convs)

        convs = layers.Conv2D(
            n_output_plane,
            (3, 3),
            strides=stride,
            padding="same",
            use_bias=False,
            kernel_initializer=INIT
        )(convs)

    convs = layers.BatchNormalization()(convs)
    convs = layers.Activation("relu")(convs)

    convs = layers.Conv2D(
        n_output_plane,
        (3, 3),
        strides=1,
        padding="same",
        use_bias=False,
        kernel_initializer=INIT
    )(convs)

    return layers.Add()([convs, shortcut])


def get_network():
    n = (DEPTH - 4) // 6
    stages = [16, 16 * WIDTH_MULT, 32 * WIDTH_MULT, 64 * WIDTH_MULT]
    inputs = keras.Input(shape=(32, 32, 3))

    # x = layers.Rescaling(1.0 / 255)(inputs)

    x = layers.Conv2D(
        stages[0],
        (3, 3),
        padding="same",
        use_bias=False,
        kernel_initializer=INIT,
    )(inputs)

    for i in range(1, 4):
        x = wide_basic(x, stages[i - 1], stages[i], stride=(1 if i == 1 else 2))
        for _ in range(n - 1):
            x = wide_basic(x, stages[i], stages[i], stride=1)

    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.GlobalAveragePooling2D()(x)

    outputs = layers.Dense(
        10,
    )(x)

    return keras.Model(inputs, outputs)


class KaimingNormal(VarianceScaling):
	def __init__(self, seed=None):
		super(KaimingNormal, self).__init__(
				scale=2., mode='fan_in', distribution='untruncated_normal', seed=seed)

	def get_config(self):
		return {'seed': self.seed}

seed = None
np_rng = np.random.RandomState(seed=seed)
dropout = 0.4

def _randint():
    return np_rng.randint(0, 2**16)

def wideresnet(input_shape, n_classes):
    depth = 28
    width = 10
    assert ((depth - 4) % 6 == 0)
    n = (depth - 4) // 6
    n_stages = [16, 16 * width, 32 * width, 64 * width]

    def weight_init():
        return KaimingNormal(seed=_randint())

    def conv(net, n, size=(1, 1), stride=(1, 1), padding=(1, 1)):
        if padding is not None:
            net = layers.ZeroPadding2D(padding=padding)(net)
        return layers.Conv2D(n, size, strides=stride, padding='valid',
                      kernel_initializer=weight_init(),
                      use_bias=False)(net)

    def bn(net):
        return layers.BatchNormalization(momentum=0.9, epsilon=1e-5,
                                  gamma_initializer='uniform')(net)

    def wideresnet_unit(net, n, stride=(1, 1), shortcut=False):
        unit = bn(net)
        unit = layers.Activation('relu')(unit)
        conv_a = conv(unit, n, size=(3, 3), stride=stride)
        conv_a = bn(conv_a)
        conv_a = layers.Activation('relu')(conv_a)
        if dropout > 0:
            conv_a = layers.Dropout(dropout, seed=_randint())(conv_a)
        conv_a = conv(conv_a, n, size=(3, 3), stride=(1, 1))
        if shortcut:
            conv_b = conv(unit, n, size=(1, 1), stride=stride, padding=None)
        else:
            conv_b = net
        unit = layers.Add()([conv_a, conv_b])
        return unit

    def wideresnet_stack(net, n, height, stride):
        stack = wideresnet_unit(net, n, stride, shortcut=True)
        for i in range(height - 1):
            stack = wideresnet_unit(stack, n)
        return stack

    in_layer = layers.Input(shape=input_shape)
    conv1 = conv(in_layer, n_stages[0], size=(3, 3), stride=(1, 1))
    conv2 = wideresnet_stack(conv1, n_stages[1], n, (1, 1))
    conv3 = wideresnet_stack(conv2, n_stages[2], n, (2, 2))
    conv4 = wideresnet_stack(conv3, n_stages[3], n, (2, 2))
    wrn = bn(conv4)
    wrn = layers.Activation('relu')(wrn)
    wrn = layers.AveragePooling2D(pool_size=(8, 8), strides=(1, 1), padding='valid')(wrn)
    wrn = layers.Flatten()(wrn)
    wrn = layers.Dense(n_classes, kernel_initializer=weight_init())(wrn)
    model = tf.keras.Model(inputs=in_layer, outputs=wrn)
    return model