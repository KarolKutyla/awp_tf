import tensorflow as tf

x = tf.Variable(2.0)
y = tf.Variable(3.0)
a = tf.Variable(0.0)

with tf.GradientTape() as tape:
    x_sq = x * x

    # Temporarily stop recording to prevent interference
    with tape.stop_recording():
        a.assign(y * y + 1)  # This op is ignored by the tape
    a_sq = 5 * a
    # Resume or proceed with normal operations
    z = x_sq + y + a_sq

# Calculate gradient
grads = tape.gradient(a_sq, [a])
print("dx:", grads[0].numpy())  # 2*x = 4.0
# print("dy:", grads[1].numpy())
# print("da:", grads[2].numpy())