import tensorflow as tf
import numpy as np

from tensorflow.keras import layers


class Aggregator(tf.keras.Model):
    def __init__(self):
        super(Aggregator, self).__init__()

    def call(self, representations):
        return tf.reduce_mean(representations, axis=1)


class Encoder(tf.keras.Model):
    def __init__(self, sizes):
        super(Encoder, self).__init__()

        self._layers = [layers.Dense(size, activation='relu')
                        for size in sizes]

    def call(self, x, y):
        out = tf.concat([x, y], axis=-1)

        for i in range(len(self._layers)):
            out = self._layers[i](out)

        return out


class Decoder(tf.keras.Model):
    def __init__(self, sizes):
        super(Decoder, self).__init__()

        self._layers = [layers.Dense(size, activation='relu')
                        for size in sizes[:-1]]
        self._layers.append(layers.Dense(sizes[-1]))

    def call(self, representation, x):
        representation = tf.tile(tf.expand_dims(
            representation, 1), [1, tf.shape(x)[1], 1])
        out = tf.concat([representation, x], axis=-1)

        for i in range(len(self._layers)):
            out = self._layers[i](out)

        mu, log_sigma = tf.split(out, 2, axis=-1)

        sigma = 0.1 + 0.9 * tf.math.softplus(log_sigma)

        return mu, sigma


class CNP(tf.keras.Model):
    def __init__(self, encoder_sizes, decoder_sizes):
        super(CNP, self).__init__()
        self.encoder = Encoder(encoder_sizes)
        self.decoder = Decoder(decoder_sizes)
        self.aggregator = Aggregator()

    def call(self, context_x, context_y, target_x):
        representations = self.encoder(context_x, context_y)
        representation = self.aggregator(representations)

        mu, sigma = self.decoder(representation, target_x)

        return mu, sigma
