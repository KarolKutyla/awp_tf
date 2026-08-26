from math import sqrt

import tensorflow as tf
from tensorflow import keras


def select_evenly(model: keras.Model) -> tuple[float, ...]:
    scales = [1.0 if var.name == 'kernel' else 0.0 for var in model.trainable_variables]
    return tuple(scales)

def select_respectable_to_variable_sizes(model: keras.Model) -> tuple[float, ...]:
    sum_of_square_roots = sum([sqrt(int(tf.size(var.value))) for var in model.trainable_variables if var.name == 'kernel'])
    number_of_kernels = len([var for var in model.trainable_variables if var.name == 'kernel'])
    scales = [
        float(sqrt(int(tf.size(var.value))) / sum_of_square_roots * number_of_kernels) if var.name == 'kernel'
        else 0.0
        for var in model.trainable_variables
    ]
    return tuple(scales)


def get_trainable_params_dict(model: keras.Model) -> dict[tuple, tf.Variable]:
    params = {}
    for layer in model.layers:
        if layer.trainable_variables:
            for var in layer.trainable_variables:
                params[(layer, var)] = 0.0
    return params


def translate_param_dict_to_scales_list(model: keras.Model, params_dict: dict[tuple, float]) -> tuple[float, ...]:
    trainable_variables_sizes = []
    for layer in model.layers:
        if layer.trainable_variables:
            for var in layer.trainable_variables:
                value = params_dict[(layer, var)]
                if value is not None:
                    trainable_variables_sizes.append(value)
    return tuple(trainable_variables_sizes)