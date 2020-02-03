import tensorflow as tf
import tensorflow_probability as tfp

from tensorflow.keras import layers, optimizers, datasets

import numpy as np

from absl import flags
from absl import app

from models import CNP
from data import MNISTDataGen

FLAGS = flags.FLAGS

flags.DEFINE_integer("batch_size", 64, "Batch size for training steps")
flags.DEFINE_integer("iter", int(2e5), "Number of iterations for training")
flags.DEFINE_integer("max_num_context", 10, "Biggest context set")

flags.DEFINE_float("lr", 1e-4, "Learning rate")

flags.DEFINE_list("encoder_arch", [128, 128, 128, 128], "Encoder architecture")
flags.DEFINE_list("decoder_arch", [128, 128, 2], "Decoder architecture")


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

        if it % 100 == 0:
            tf.print("Iteration", it, "loss", loss)


def main(argv):
    del argv
    iterations = tf.constant(FLAGS.iter)

    dataset_train = MNISTDataGen(FLAGS.batch_size, FLAGS.max_num_context)

    encoder_sizes = [int(x) for x in FLAGS.encoder_arch]
    decoder_sizes = [int(x) for x in FLAGS.decoder_arch]

    assert(decoder_sizes[-1] == 2)

    model = CNP(FLAGS.encoder_arch, FLAGS.decoder_arch)
    optimizer = optimizers.Adam(FLAGS.lr)

    train(model, optimizer, iterations, dataset_train)


if __name__ == "__main__":
    app.run(main)
