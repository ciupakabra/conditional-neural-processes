import tensorflow as tf
import tensorflow_probability as tfp

from tensorflow.keras import layers, optimizers

import numpy as np

from absl import flags
from absl import app

FLAGS = flags.FLAGS

flags.DEFINE_integer("batch_size", 64, "Batch size for training steps")
flags.DEFINE_integer("iter", int(2e5), "Number of iterations for training")
flags.DEFINE_integer("x_dims", 1, "Dimensionality of the feature vector")
flags.DEFINE_integer("y_dims", 1, "Dimensionality of the output vector")
flags.DEFINE_integer("max_num_context", 10, "Biggest context set")

flags.DEFINE_float("lr", 1e-4, "Learning rate")
flags.DEFINE_float("amplitude", 1.0, "Amplitude of the Square-Exp GP kernel")
flags.DEFINE_float("length_scale", 0.4,
                   "Length scale of the Square-Exp GP kernel")

flags.DEFINE_list("encoder_arch", [128, 128, 128, 128], "Encoder architecture")
flags.DEFINE_list("decoder_arch", [128, 128, 2], "Decoder architecture")


class GPDataGen():

    def __init__(
            self,
            batch_size,
            max_num_context,
            kernel,
            x_size=1,
            y_size=1,):

        self.max_num_context = max_num_context
        self.batch_size = batch_size
        self.kernel = kernel
        self.x_size = x_size
        self.y_size = y_size

    def make_batch(self):
        num_context = tf.random.uniform(
            (), minval=3, maxval=self.max_num_context, dtype=tf.int32)

        num_target = tf.random.uniform(
            (), minval=2, maxval=self.max_num_context, dtype=tf.int32)
        num_total_points = num_context + num_target

        x_values = tf.random.uniform(
            (self.batch_size, num_total_points, self.x_size), -2, 2)

        matrix = self.kernel.matrix(
            x_values, x_values) + 1e-4 * tf.eye(num_total_points)

        cholesky = tf.cast(tf.linalg.cholesky(
            tf.cast(matrix, tf.float64)), tf.float32)
        y_values = tf.matmul(
            tf.tile(tf.expand_dims(cholesky, 1), (1, self.y_size, 1, 1)),
            tf.random.normal((self.batch_size, self.y_size, num_total_points, 1)))

        y_values = tf.transpose(tf.squeeze(y_values, 3), [0, 2, 1])

        context_x = x_values[:, :num_context, :]
        context_y = y_values[:, :num_context, :]

        return (context_x, context_y), (x_values, y_values)


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


def loss_fun(mu, sigma, target_y):
    log_prob = tfp.distributions.MultivariateNormalDiag(
        loc=mu, scale_diag=sigma).log_prob(target_y)
    return -tf.reduce_mean(log_prob)


def train_one_step(model, optimizer, batch):

    (context_x, context_y), (target_x, target_y) = batch

    with tf.GradientTape() as tape:
        mu, sigma = model(context_x, context_y, target_x)

        loss = loss_fun(mu, sigma, target_y)

    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    return loss


@tf.function
def train(model, optimizer, iterations, dataset_train):
    for it in range(iterations):
        batch = dataset_train.make_batch()
        loss = train_one_step(model, optimizer, batch)
        if it % 1000 == 0:
            tf.print("Iteration", it, "loss", loss)


def main(argv):
    del argv
    iterations = tf.constant(FLAGS.iter)

    kernel = tfp.math.psd_kernels.ExponentiatedQuadratic(
        amplitude=FLAGS.amplitude, length_scale=FLAGS.length_scale)

    dataset_train = GPDataGen(FLAGS.batch_size, FLAGS.max_num_context, kernel)

    encoder_sizes = [int(x) for x in FLAGS.encoder_arch]
    decoder_sizes = [int(x) for x in FLAGS.decoder_arch]

    assert(decoder_sizes[-1] == 2 * FLAGS.y_dims)

    model = CNP(FLAGS.encoder_arch, FLAGS.decoder_arch)
    optimizer = optimizers.Adam(FLAGS.lr)

    train(model, optimizer, iterations, dataset_train)


if __name__ == "__main__":
    app.run(main)
