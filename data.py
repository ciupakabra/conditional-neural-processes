import tensorflow as tf
import numpy as np

from tensorflow.keras import datasets


class MNISTDataGen():
    def __init__(self, batch_size, max_num_context, testing=False):
        (X_train, _), (X_test, _) = datasets.mnist.load_data()

        if testing:
            self.X = X_test
        else:
            self.X = X_train

        self.X = self.X.astype("float32").reshape(-1, 28 * 28, 1) / 255.

        self.x_values = np.indices((28, 28)).transpose(
            [1, 2, 0]).reshape((-1, 2)).astype("float32")
        self.x_values = np.tile(self.x_values, (batch_size, 1, 1))
        self.x_values = tf.constant(self.x_values)

        self.X = tf.constant(self.X)

        self.x_values = tf.constant(self.x_values)

        self.batch_size = batch_size
        self.max_num_context = max_num_context

    def make_batch(self):
        img_idxs = tf.random.uniform(
            (self.batch_size,), minval=0, maxval=self.X.shape[0], dtype=tf.int32)

        y_values = tf.gather(self.X, img_idxs, axis=0)

        num_context = tf.random.uniform(
            (), minval=3, maxval=self.max_num_context, dtype=tf.int32)

        idxs = tf.random.shuffle(range(28 * 28))

        context_x = tf.gather(self.x_values, idxs[:num_context], axis=1)
        context_y = tf.gather(y_values, idxs[:num_context], axis=1)

        return (context_x, context_y), (self.x_values, y_values)


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
