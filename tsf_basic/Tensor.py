import tensorflow as tf

rank_4_tensor = tf.zeros([3, 2, 4, 5])
print("Type of every element:", rank_4_tensor.dtype)
print("Number of axes:", rank_4_tensor.ndim)
print("Shape of tensor:", rank_4_tensor.shape)
print("Elements along axis 0 of tensor:", rank_4_tensor.shape[0])
print("Elements along the last axis of tensor:", rank_4_tensor.shape[-1])
print("Total number of elements (3*2*4*5): ", tf.size(rank_4_tensor).numpy())

# Index
print("Everything:", rank_4_tensor[:].numpy())
print("Before 4:", rank_4_tensor[:4].numpy())
print("From 4 to the end:", rank_4_tensor[4:].numpy())
print("From 2, before 7:", rank_4_tensor[2:7].numpy())
print("Every other item:", rank_4_tensor[::2].numpy())
print("Reversed:", rank_4_tensor[::-1].numpy())

# Pull out a single value
print(rank_4_tensor[1, 1].numpy())

# Get row and column tensors
print("Second row:", rank_4_tensor[1, :].numpy())
print("Second column:", rank_4_tensor[:, 1].numpy())
print("Last row:", rank_4_tensor[-1, :].numpy())
print("First item in last column:", rank_4_tensor[0, -1].numpy())
print("Skip the first row:")
print(rank_4_tensor[1:, :].numpy(), "\n")


# Ragged Tensor
digits = tf.ragged.constant([[3, 1, 4, 1], [], [5, 9, 2], [6], []])
words = tf.ragged.constant([['So', 'Long'], ['Thank', 'For', 'The', 'Food']])
print(tf.add(digits, 3))
print(tf.reduce_mean(digits, axis=1))
print(tf.concat([digits, [[5, 3]]], axis=0))
print(tf.tile(digits, [1, 2]))
print(tf.strings.substr(words, 0, 2))
print(tf.map_fn(tf.math.square, digits))



