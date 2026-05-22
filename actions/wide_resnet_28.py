from tensorflow import keras
from tensorflow.keras import layers

RESIZE_TO = 32

SOURCE_BATCH_SIZE = 64
TARGET_BATCH_SIZE = 3 * SOURCE_BATCH_SIZE  # Reference: Section 3.2
EPOCHS = 2

LEARNING_RATE = 0.03

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
            kernel_initializer=INIT,
            kernel_regularizer=keras.regularizers.l2(WEIGHT_DECAY),
        )(x)

        convs = layers.Conv2D(
            n_output_plane,
            (3, 3),
            strides=stride,
            padding="same",
            use_bias=False,
            kernel_initializer=INIT,
            kernel_regularizer=keras.regularizers.l2(WEIGHT_DECAY),
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
            kernel_initializer=INIT,
            kernel_regularizer=keras.regularizers.l2(WEIGHT_DECAY),
        )(convs)

    convs = layers.BatchNormalization()(convs)
    convs = layers.Activation("relu")(convs)

    convs = layers.Conv2D(
        n_output_plane,
        (3, 3),
        strides=1,
        padding="same",
        use_bias=False,
        kernel_initializer=INIT,
        kernel_regularizer=keras.regularizers.l2(WEIGHT_DECAY),
    )(convs)

    return layers.Add()([convs, shortcut])


def get_network():
    n = (DEPTH - 4) // 6
    stages = [16, 16 * WIDTH_MULT, 32 * WIDTH_MULT, 64 * WIDTH_MULT]
    inputs = keras.Input(shape=(32, 32, 3))

    x = layers.Rescaling(1.0 / 255)(inputs)

    x = layers.Conv2D(
        stages[0],
        (3, 3),
        padding="same",
        use_bias=False,
        kernel_initializer=INIT,
        kernel_regularizer=keras.regularizers.l2(WEIGHT_DECAY),
    )(x)

    for i in range(1, 4):
        x = wide_basic(x, stages[i - 1], stages[i], stride=(1 if i == 1 else 2))
        for _ in range(n - 1):
            x = wide_basic(x, stages[i], stages[i], stride=1)

    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.GlobalAveragePooling2D()(x)

    outputs = layers.Dense(
        10,
        kernel_regularizer=keras.regularizers.l2(WEIGHT_DECAY),
    )(x)

    return keras.Model(inputs, outputs)