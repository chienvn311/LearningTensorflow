from abc import ABC
import tensorflow as tf


class ResnetIdentityBlock(tf.keras.Model, ABC):
    def __init__(self, kernel_size, filters):
        super(ResnetIdentityBlock, self).__init__(name='')
        filters1, filters2, filters3 = filters

        self.conv2a = tf.keras.layers.Conv2D(filters1, (1, 1))
        self.bn2a = tf.keras.layers.BatchNormalization()

        self.conv2b = tf.keras.layers.Conv2D(filters2, kernel_size, padding='same')
        self.bn2b = tf.keras.layers.BatchNormalization()

        self.conv2c = tf.keras.layers.Conv2D(filters3, (1, 1))
        self.bn2c = tf.keras.layers.BatchNormalization()

    def call(self, inputs, training=None, mask=None):
        x = self.conv2a(inputs)
        x = self.bn2a(x, training=training)
        x = tf.nn.relu(x)

        x = self.conv2b(x)
        x = self.bn2b(x, training=training)
        x = tf.nn.relu(x)

        x = self.conv2c(x)
        x = self.bn2c(x, training=training)

        x += inputs
        return tf.nn.relu(x)
    # def call(self, input_tensor, training=False):
    #     x = self.conv2a(input_tensor)
    #     x = self.bn2a(x, training=training)
    #     x = tf.nn.relu(x)
    #
    #     x = self.conv2b(x)
    #     x = self.bn2b(x, training=training)
    #     x = tf.nn.relu(x)
    #
    #     x = self.conv2c(x)
    #     x = self.bn2c(x, training=training)
    #
    #     x += input_tensor
    #     return tf.nn.relu(x)


block = ResnetIdentityBlock(1, [1, 2, 3])

# Another way to do this
my_seq = tf.keras.Sequential([tf.keras.layers.Conv2D(1, (1, 1),
                                                     input_shape=(
                                                         None, None, 3)),
                              tf.keras.layers.BatchNormalization(),
                              tf.keras.layers.Conv2D(2, 1,
                                                     padding='same'),
                              tf.keras.layers.BatchNormalization(),
                              tf.keras.layers.Conv2D(3, (1, 1)),
                              tf.keras.layers.BatchNormalization()])
my_seq(tf.zeros([1, 2, 3, 3]))
