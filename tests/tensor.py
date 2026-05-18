import tensorflow as tf
import numpy as np

def indices(array):
    keep_dim = tf.reshape(tf.constant(tf.shape(array)[0]), 1)
    as_one = tf.ones(tf.size(tf.shape(array)[1:]), dtype=tf.int32)
    return tf.concat([keep_dim, as_one], axis=0)

array = np.array(
    [
        [[0.2, 0.2], [0.2, 0.2]],
        [[1.0, 1.0], [1.0, 1.0]],
        [[2.0, 2.0], [2.0, 2.0]],
        [[1.0, 2.0], [3.0, 4.0]],
    ]
)
array = tf.constant(array)

norm_indices = tuple(range(tf.rank(array))[1:])
print(norm_indices)
array_norm = tf.norm(array, axis = norm_indices)
scaled_norm = tf.minimum(1.0, array_norm)


print(array_norm)
print(tf.norm(array, axis=norm_indices, keepdims=True))

# print(indices(array))
#
# print(tf.reshape(scaled_norm, indices(array)) * array)

