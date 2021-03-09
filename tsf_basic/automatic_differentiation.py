import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


# x = tf.Variable(3.0)
# with tf.GradientTape() as tape:
#     y = x**2
# # dy = 2x * dx
# dy_dx = tape.gradient(y, x)
# print(dy_dx.numpy())
#
#
# layer = tf.keras.layers.Dense(2, activation='relu')
# x = tf.constant([[1., 2., 3.]])
#
# with tf.GradientTape() as tape:
#     tape.watch(x)
#     # Forward pass
#     y = layer(x)
#     loss = tf.reduce_mean(y**2)
#
# # Calculate gradients with respect to every trainable variable
# grad = tape.gradient(loss, layer.trainable_variables)
#
# for var, g in zip(layer.trainable_variables, grad):
#     print(f'{var.name}, shape: {g.shape}')


x = tf.Variable(2.0)
y = tf.Variable(3.0)
with tf.GradientTape() as t:
    x_sq = x * x
    y_sq = y * y
    z = x_sq + y_sq
grad = t.gradient(z, {'x': x, 'y': y})

print('dz/dx:', grad['x']) # 2*x => 4
print('dz/dy:', grad['y']) # 2*y => 6

############################################################
# Stop recording
with tf.GradientTape() as t:
    x_sq = x * x
    with t.stop_recording():
        y_sq = y * y
    z = x_sq + y_sq

grad = t.gradient(z, {'x': x, 'y': y})

print('dz/dx:', grad['x']) # 2*x => 4
print('dz/dy:', grad['y']) # None


############################################################
# Reset
reset = True
with tf.GradientTape() as t:
  y_sq = y * y
  if reset:
    # Throw out all the tape recorded so far
    t.reset()
  z = x * x + y_sq

grad = t.gradient(z, {'x': x, 'y': y})

print('dz/dx:', grad['x'])  # 2*x => 4
print('dz/dy:', grad['y']) # None


####################################################
# Stop gradient
with tf.GradientTape() as t:
  y_sq = y**2
  z = x**2 + tf.stop_gradient(y_sq)

grad = t.gradient(z, {'x': x, 'y': y})

print('dz/dx:', grad['x'])  # 2*x => 4
print('dz/dy:', grad['y'])


####################################################
# Custom
@tf.custom_gradient
def clip_gradients(y):
    def backward(dy):
        return tf.clip_by_norm(dy, 0.5)
    return y, backward

v = tf.Variable(2.0)
with tf.GradientTape() as t:
    output = clip_gradients(v * v)
print(t.gradient(output, v))  # calls "backward", which clips 4 to 2


####################################################
# Multiple
x0 = tf.constant(0.0)
x1 = tf.constant(0.0)

with tf.GradientTape() as tape0, tf.GradientTape() as tape1:
    tape0.watch(x0)
    tape1.watch(x1)
    y0 = tf.math.sin(x0)
    y1 = tf.nn.sigmoid(x1)
    y = y0 + y1
    ys = tf.reduce_sum(y)

tape0.gradient(ys, x0).numpy()   # cos(x) => 1.0
tape1.gradient(ys, x1).numpy()   # sigmoid(x1)*(1-sigmoid(x1)) => 0.25

# Higher order
x = tf.Variable(1.0)  # Create a Tensorflow variable initialized to 1.0

with tf.GradientTape() as t2:
    with tf.GradientTape() as t1:
        y = x * x * x
    # Compute the gradient inside the outer `t2` context manager
    # which means the gradient computation is differentiable as well.
    dy_dx = t1.gradient(y, x)
d2y_dx2 = t2.gradient(dy_dx, x)

print('dy_dx:', dy_dx.numpy())  # 3 * x**2 => 3.0
print('d2y_dx2:', d2y_dx2.numpy())  # 6 * x => 6.0