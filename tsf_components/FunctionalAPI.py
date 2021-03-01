import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

inputs = keras.Input(shape=(784,))
print(inputs.shape)
print(inputs.dtype)

dense = layers.Dense(64, activation="relu")
x = dense(inputs)


# ------------------------------------------------------
keras_inputs = keras.Input((28, 28, 2), name='image')
x = layers.Conv2D(16, 3, activation='relu')(keras_inputs)
x = layers.Conv2D(32, 3, activation='rely')(x)
x = layers.MaxPool2D(3)(x)
x = layers.Conv2D(32, 3, activation='rely')(x)
x = layers.Conv2D(16, 3, activation='rely')(x)
keras_outputs = layers.GlobalAvgPool2D()(x)
model = keras.Model(keras_inputs, keras_outputs, name='model')
model.summary()
# Plot the model as a graph
keras.utils.plot_model(model, "my_first_model.png")
keras.utils.plot_model(model, "my_first_model_with_shape_info.png", show_shapes=True)

x = layers.Reshape((4, 4, 1))(keras_outputs)
# Reverse of a Conv2D layer is a Conv2DTranspose layer
x = layers.Conv2DTranspose(16, 3, activation="relu")(x)
x = layers.Conv2DTranspose(32, 3, activation="relu")(x)
# reverse of a MaxPooling2D layer is an UpSampling2D
x = layers.UpSampling2D(3)(x)
x = layers.Conv2DTranspose(16, 3, activation="relu")(x)
keras_outputs2 = layers.Conv2DTranspose(1, 3, activation="relu")(x)
model_2 = keras.Model(keras_inputs, keras_outputs2, name="autoencoder")
model_2.summary()


# -----------------------------------------------------
# Nested Model
def get_model():
    input = keras.Input(shape=(128,))
    output = layers.Dense(1)(inputs)
    return keras.Model(input, output)


model_1 = get_model()
model_2 = get_model()
model_3 = get_model()

inputs = keras.Input(shape=(128,))
l1 = model_1(inputs)


# -------------------------------------------------------
# Model with multiple inputs and outputs
num_tags = 12  # Number of unique issue tags
num_words = 10000  # Size of vocabulary obtained when preprocessing text data
num_departments = 4  # Number of departments for predictions

title_input = keras.Input(
    shape=(None,), name="title"
)  # Variable-length sequence of ints
body_input = keras.Input(shape=(None,), name="body")  # Variable-length sequence of ints
tags_input = keras.Input(
    shape=(num_tags,), name="tags"
)  # Binary vectors of size `num_tags`

# Embed each word in the title into a 64-dimensional vector
title_features = layers.Embedding(num_words, 64)(title_input)
# Embed each word in the text into a 64-dimensional vector
body_features = layers.Embedding(num_words, 64)(body_input)

# Reduce sequence of embedded words in the title into a single 128-dimensional vector
title_features = layers.LSTM(128)(title_features)
# Reduce sequence of embedded words in the body into a single 32-dimensional vector
body_features = layers.LSTM(32)(body_features)

# Merge all available features into a single large vector via concatenation
x = layers.concatenate([title_features, body_features, tags_input])

# Stick a logistic regression for priority prediction on top of the features
priority_pred = layers.Dense(1, name="priority")(x)
# Stick a department classifier on top of the features
department_pred = layers.Dense(num_departments, name="department")(x)

# Instantiate an end-to-end model predicting both priority and department
model = keras.Model(
    inputs=[title_input, body_input, tags_input],
    outputs=[priority_pred, department_pred],
)
keras.utils.plot_model(model, "multi_input_and_output_model.png", show_shapes=True)
# Assign different losses to each output
model.compile(
    optimizer=keras.optimizers.RMSprop(1e-3),
    loss={
        "priority": keras.losses.BinaryCrossentropy(from_logits=True),
        "department": keras.losses.CategoricalCrossentropy(from_logits=True),
    },
    loss_weights=[1.0, 0.2],
)

# Dummy input data
title_data = np.random.randint(num_words, size=(1280, 10))
body_data = np.random.randint(num_words, size=(1280, 100))
tags_data = np.random.randint(2, size=(1280, num_tags)).astype("float32")

# Dummy target data
priority_targets = np.random.random(size=(1280, 1))
dept_targets = np.random.randint(2, size=(1280, num_departments))

model.fit(
    {"title": title_data, "body": body_data, "tags": tags_data},
    {"priority": priority_targets, "department": dept_targets},
    epochs=2,
    batch_size=32,
)

# ----------------------------------------------------------------
# Non-linear connectivity topologies
inputs = keras.Input(shape=(32, 32, 3), name="img")
x = layers.Conv2D(32, 3, activation="relu")(inputs)
x = layers.Conv2D(64, 3, activation="relu")(x)
block_1_output = layers.MaxPooling2D(3)(x)

x = layers.Conv2D(64, 3, activation="relu", padding="same")(block_1_output)
x = layers.Conv2D(64, 3, activation="relu", padding="same")(x)
block_2_output = layers.add([x, block_1_output])

x = layers.Conv2D(64, 3, activation="relu", padding="same")(block_2_output)
x = layers.Conv2D(64, 3, activation="relu", padding="same")(x)
block_3_output = layers.add([x, block_2_output])

x = layers.Conv2D(64, 3, activation="relu")(block_3_output)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(256, activation="relu")(x)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(10)(x)

model = keras.Model(inputs, outputs, name="toy_resnet")
model.summary()
keras.utils.plot_model(model, "mini_resnet.png", show_shapes=True)

(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

model.compile(
    optimizer=keras.optimizers.RMSprop(1e-3),
    loss=keras.losses.CategoricalCrossentropy(from_logits=True),
    metrics=["acc"],
)
# We restrict the data to the first 1000 samples so as to limit execution time
# on Colab. Try to train on the entire dataset until convergence!
model.fit(x_train[:1000], y_train[:1000], batch_size=64, epochs=1, validation_split=0.2)


# ----------------------------------------------------------------
# Shared layer
