import tensorflow as tf
import timeit
from datetime import datetime


# Define a Python function.
# def a_regular_function(x, y, b):
#   x = tf.matmul(x, y)
#   x = x + b
#   return x
#
# # `a_function_that_uses_a_graph` is a TensorFlow `Function`.
# a_function_that_uses_a_graph = tf.function(a_regular_function)
#
# # Make some tensors.
# x1 = tf.constant([[1.0, 2.0]])
# y1 = tf.constant([[2.0], [3.0]])
# b1 = tf.constant(4.0)
#
# orig_value = a_regular_function(x1, y1, b1).numpy()
# # Call a `Function` like a Python function.
# tf_function_value = a_function_that_uses_a_graph(x1, y1, b1).numpy()
# assert(orig_value == tf_function_value)


def inner_function(x, y, b):
  x = tf.matmul(x, y)
  x = x + b
  return x

# Use the decorator to make `outer_function` a `Function`.
@tf.function
def outer_function(x):
  y = tf.constant([[2.0], [3.0]])
  b = tf.constant(4.0)

  return inner_function(x, y, b)

# Note that the callable will create a graph that
# includes `inner_function` as well as `outer_function`.
outer_function(tf.constant([[1.0, 2.0]])).numpy()
