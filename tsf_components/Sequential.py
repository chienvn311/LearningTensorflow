import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


layer_1 = layers.Dense(2, activation="relu", name="1")
layer_2 = layers.Dense(3, activation="relu", name="2")
layer_3 = layers.Dense(4, activation="relu", name="3")
model_layers = [layer_1, layer_2, layer_3]

# Define Sequential model
model = keras.Sequential(layers=model_layers)

# No weights at this stage!
# At this point, you can't do this:
# model.weights
# model.summary()

# Call model
x = tf.ones((3, 3))
y = model(x)

# Weights have created, built
print("Number of weights after calling the model:", len(model.weights))
model.summary()


# -------------------------------------------------------------------
model2 = keras.Sequential()
# Add default input
model2.add(keras.Input(shape=(4,)))
# Pass an input_shape argument to layer
model.add(layers.Dense(2, activation="relu", input_shape=(4,)))
# Remove layer
model2.pop()
# Alternative way to add layers
model2.add(layer_1)
model2.add(layer_2)
model2.add(layer_3)
# Summary without building model
model2.summary()
