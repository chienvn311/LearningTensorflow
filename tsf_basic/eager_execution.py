import tensorflow as tf

# tf.executing_eagerly()
# x = [[2.]]
# m = tf.matmul(x, x)
# tf.add()
# print(m)


w = tf.Variable([[1.0]])
with tf.GradientTape() as tape:
    loss = w * w
grad = tape.gradient(target=loss, sources=w)
print(grad)

