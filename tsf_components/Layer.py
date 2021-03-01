import tensorflow as tf


print(tf.test.is_gpu_available)

# Layers are objects
# To construct a layer, simply construct the object.
# Most layers take as a first argument the number of output dimensions / channels.
layer = tf.keras.layers.Dense(100, input_shape=(None, 5))
layer(tf.zeros([10, 5]))

# Inspect all variables
layer.variables
layer.kernel
layer.bias


# Standard Custom Layer
class MyCustomLayer(tf.keras.layers.Layer):
    # Layer's weight
    def __init__(self, num_outputs):
        super(MyCustomLayer, self).__init__()
        self.num_outputs = num_outputs

    # __call__() method of your layer will automatically run build the first time it is called
    def built(self, input_shape):
        self.kernel = self.add_weight("kernel",
                                      shape=[int(input_shape[-1]),
                                             self.num_outputs])

    # Layer pass forward
    def call(self, inputs, **kwargs):
        return tf.matmul(input, self.kernel)


custom_layer = MyCustomLayer(10)


# --------------------------------------------------------------------------
# Non trainable weight
class ComputeSum(tf.keras.layers.Layer):
    def __init__(self, input_dim):
        super(ComputeSum, self).__init__()
        self.total = tf.Variable(initial_value=tf.zeros((input_dim,)), trainable=False)
        # self.total = tf.Variable(initial_value=tf.zeros((input_dim,)))

    def call(self, inputs):
        self.total.assign_add(tf.reduce_sum(inputs, axis=0))
        return self.total


z = tf.zeros((2,))
x = tf.ones((2, 2))
my_sum = ComputeSum(2)
y = my_sum(x)
print(y.numpy())
y1 = my_sum(x)
print(y.numpy())
print(y1.numpy())

print("weights:", len(my_sum.weights))
print("non-trainable weights:", len(my_sum.non_trainable_weights))

# It's not included in the trainable weights:
print("trainable_weights:", my_sum.trainable_weights)


# ------------------------------------------------------------------------
# Deferring weight creation until the shape of the inputs is known

class Linear(tf.keras.layers.Layer):
    def __init__(self, units=32):
        super(Linear, self).__init__()
        self.units = units

    def build(self, input_shape):
        self.w = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer="random_normal",
            trainable=True,
        )
        self.b = self.add_weight(
            shape=(self.units,), initializer="random_normal", trainable=True
        )

    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.b

    # Enable serialization
    def get_config(self):
        config = super(Linear, self).get_config()
        config.update({"units": self.units})
        return config

    # Deserializing the layer from its config
    def from_config(cls, config):
        return cls(**config)

layer = Linear(64)
config = layer.get_config()
print(config)
new_layer = layer.from_config(config)



# Assign a Layer instance as attribute of another Layer
# Let's assume we are reusing the Linear class
# with a `build` method that we defined above.
class MLPBlock(tf.keras.layers.Layer):
    def __init__(self):
        super(MLPBlock, self).__init__()
        self.linear_1 = Linear(32)
        self.linear_2 = Linear(32)
        self.linear_3 = Linear(1)

    def call(self, inputs):
        x = self.linear_1(inputs)
        x = tf.nn.relu(x)
        x = self.linear_2(x)
        x = tf.nn.relu(x)
        return self.linear_3(x)


mlp = MLPBlock()
y = mlp(tf.ones(shape=(3, 64)))  # The first call to the `mlp` will create the weights
print("weights:", len(mlp.weights))
print("trainable weights:", len(mlp.trainable_weights))


# ---------------------------------------------------------------------
# Metric and loss
# Consider the following layer: a "logistic endpoint" layer. It takes as inputs predictions & targets,
# it computes a loss which it tracks via add_loss(), and it computes an accuracy scalar,
# which it tracks via add_metric().
class LogisticEndpoint(tf.keras.layers.Layer):
    def __init__(self, name=None):
        super(LogisticEndpoint, self).__init__(name=name)
        self.loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        self.accuracy_fn = tf.keras.metrics.BinaryAccuracy()

    def call(self, targets, logits, sample_weights=None):
        # Compute the training-time loss value and add it
        # to the layer using `self.add_loss()`.
        loss = self.loss_fn(targets, logits, sample_weights)
        self.add_loss(loss)

        # Log accuracy as a metric and add it
        # to the layer using `self.add_metric()`.
        acc = self.accuracy_fn(targets, logits, sample_weights)
        self.add_metric(acc, name="accuracy")

        # Return the inference-time prediction tensor (for `.predict()`).
        return tf.nn.softmax(logits)


layer = LogisticEndpoint()

targets = tf.ones((2, 2))
logits = tf.ones((2, 2))
y = layer(targets, logits)

print(layer.losses)
print("layer.metrics:", layer.metrics)
print("current accuracy value:", float(layer.metrics[0].result()))


# ---------------------------------------------------------------------
# Customer drop out layer
class CustomDropout(tf.keras.layers.Layer):
    def __init__(self, rate, **kwargs):
        super(CustomDropout, self).__init__(**kwargs)
        self.rate = rate

    def call(self, inputs, training=None):
        if training:
            return tf.nn.dropout(inputs, rate=self.rate)
        return inputs

