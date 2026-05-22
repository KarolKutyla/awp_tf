import tensorflow as tf
import keras_cv

from actions import preact_resnet_18
from actions import wide_resnet_28

def load_tensorflow_resnet(steps_per_epoch):
    backbone = keras_cv.models.ResNet18Backbone(
        include_rescaling=False,
        input_shape=(32, 32, 3)
    )

    x = backbone.outputs[0]
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    outputs = tf.keras.layers.Dense(10)(x)

    keras_resnet = tf.keras.Model(backbone.inputs, outputs)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
        boundaries=[100 * steps_per_epoch, 150 * steps_per_epoch],
        values=[0.1, 0.01, 0.001]
    )
    optimizer = tf.keras.optimizers.SGD(learning_rate=schedule, momentum=0.0, nesterov=False)
    keras_resnet.compile(loss=loss, optimizer=optimizer)
    optimizer.build(keras_resnet.trainable_variables)
    keras_resnet.name = "resnet_18"
    # print(keras_resnet.summary())

    return keras_resnet


def load_tensorflow_resnet_152(steps_per_epoch):
    backbone = keras_cv.models.ResNet152V2Backbone(
        include_rescaling=False,
        input_shape=(224, 224, 3)
    )

    x = backbone.outputs[0]
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    outputs = tf.keras.layers.Dense(10)(x)

    keras_resnet = tf.keras.Model(backbone.inputs, outputs)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
        boundaries=[100 * steps_per_epoch, 150 * steps_per_epoch],
        values=[0.1, 0.01, 0.001]
    )
    optimizer = tf.keras.optimizers.SGD(learning_rate=schedule, momentum=0.0, nesterov=False)
    keras_resnet.compile(loss=loss, optimizer=optimizer)
    optimizer.build(keras_resnet.trainable_variables)
    keras_resnet.name = "resnet_152v2"
    # print(keras_resnet.summary())

    return keras_resnet


def load_tensorflow_resnet_101(steps_per_epoch):
    backbone = keras_cv.models.ResNet101V2Backbone(
        include_rescaling=False,
        input_shape=(224, 224, 3)
    )

    x = backbone.outputs[0]
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    outputs = tf.keras.layers.Dense(10)(x)

    keras_resnet = tf.keras.Model(backbone.inputs, outputs)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
        boundaries=[100 * steps_per_epoch, 150 * steps_per_epoch],
        values=[0.1, 0.01, 0.001]
    )
    optimizer = tf.keras.optimizers.SGD(learning_rate=schedule, momentum=0.0, nesterov=False)
    keras_resnet.compile(loss=loss, optimizer=optimizer)
    optimizer.build(keras_resnet.trainable_variables)
    keras_resnet.name = "resnet_101v2"
    # print(keras_resnet.summary())

    return keras_resnet


def load_preact_resnet_18(steps_per_epoch):
    model = preact_resnet_18.PreActResNet18(
        input_shape=(32, 32, 3),
        num_classes=10,
        width_mult=1
    )
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
        boundaries=[100 * steps_per_epoch, 150 * steps_per_epoch],
        values=[0.1, 0.01, 0.001]
    )
    optimizer = tf.keras.optimizers.SGD(learning_rate=schedule, momentum=0.0, nesterov=False)
    model.compile(loss=loss, optimizer=optimizer)
    optimizer.build(model.trainable_variables)
    return model

def load_wide_resnet(steps_per_epoch):
    model = wide_resnet_28.get_network()
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
        boundaries=[100 * steps_per_epoch, 150 * steps_per_epoch],
        values=[0.1, 0.01, 0.001]
    )
    optimizer = tf.keras.optimizers.SGD(learning_rate=schedule, momentum=0.0, nesterov=False)
    model.compile(loss=loss, optimizer=optimizer)
    # optimizer.build(model.trainable_variables)
    model.name = "wide_resnet_28_10"
    return model

def _load_tensorflow_resnet_18_v2(steps_per_epoch):
    backbone = keras_cv.models.ResNet18V2Backbone(
        include_rescaling=False,
        input_shape=(32, 32, 3)
    )

    x = backbone.outputs[0]
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    outputs = tf.keras.layers.Dense(10)(x)

    model = tf.keras.Model(backbone.inputs, outputs)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
        boundaries=[100 * steps_per_epoch, 150 * steps_per_epoch],
        values=[0.1, 0.01, 0.001]
    )
    optimizer = tf.keras.optimizers.SGD(learning_rate=schedule, momentum=0.0, nesterov=False)
    model.compile(loss=loss, optimizer=optimizer)
    optimizer.build(model.trainable_variables)
    return model


def load_tensorflow_resnet_50_v2_for_normal_training(steps_per_epoch):
    model = _load_tensorflow_resnet_18_v2(steps_per_epoch)
    model.name = "resnet_18v2_normal"
    return model


def load_tensorflow_resnet_18_v2_for_adversarial_training(steps_per_epoch):
    model = _load_tensorflow_resnet_18_v2(steps_per_epoch)
    model.name = "resnet_18v2_adversarial"
    return model


def load_tensorflow_resnet_18_v2_for_awp_training(steps_per_epoch):
    model = _load_tensorflow_resnet_18_v2(steps_per_epoch)
    model.name = "resnet_18v2_awp"
    return model


def load_tensorflow_resnet_18_v2_for_awp_training_with_alternate_iterations(steps_per_epoch):
    model = _load_tensorflow_resnet_18_v2(steps_per_epoch)
    model.name = "resnet_18v2_awp_alternate"
    return model


def load_tensorflow_resnet_50_v2(steps_per_epoch):
    backbone = keras_cv.models.ResNet50V2Backbone(
        include_rescaling=False,
        input_shape=(224, 224, 3)
    )

    x = backbone.outputs[0]
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(256)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(10)(x)

    keras_resnet = tf.keras.Model(backbone.inputs, outputs)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
        boundaries=[100 * steps_per_epoch, 150 * steps_per_epoch],
        values=[0.1, 0.01, 0.001]
    )
    optimizer = tf.keras.optimizers.SGD(learning_rate=schedule, momentum=0.9, nesterov=False)
    keras_resnet.compile(loss=loss, optimizer=optimizer)
    optimizer.build(keras_resnet.trainable_variables)
    keras_resnet.name = "resnet_50v2"
    # print(keras_resnet.summary())

    return keras_resnet