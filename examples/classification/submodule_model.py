import tensorflow as tf


def second_dummy_model():
    inputs = tf.keras.Input((1000, ))
    x = tf.keras.layers.Dense(10)(inputs)
    x = tf.keras.layers.Dense(1000)(x)
    return tf.keras.Model(inputs, x)


class SubmoduledModel(tf.keras.Model):
    def __init__(self):
        super(SubmoduledModel, self).__init__()
        self.first_submodule = tf.keras.applications.mobilenet_v2.MobileNetV2()
        self.second_submodule = second_dummy_model()

    def call(self, inputs, training=None, mask=None):
        x = self.first_submodule(inputs)
        x = self.second_submodule(x)
        return x
