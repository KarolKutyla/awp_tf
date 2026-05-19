import tensorflow as tf
import keras_cv

from actions import preact_resnet_18

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

    # print(keras_resnet.summary())

    return keras_resnet


def load_preact_resnet_18(steps_per_epoch):
    model = preact_resnet_18.PreActResNet18(
        input_shape=(32, 32, 3),
        num_classes=10,
        width_mult=5
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