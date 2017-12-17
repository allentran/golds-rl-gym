import math

from tensorflow.contrib import keras
import tensorflow as tf


def dense_layers(X, add_summaries=False):
  fc1 = tf.contrib.layers.fully_connected(
    inputs=X,
    num_outputs=256,
    scope="fc1")

  if add_summaries:
    tf.contrib.layers.summarize_activation(fc1)

  return fc1


def rnn_graph_lstm(inputs, hidden_size, num_layers, is_training):
    def length(sequence):
        used = tf.sign(tf.reduce_max(tf.abs(sequence), 2))
        seq_length = tf.reduce_sum(used, 1)
        seq_length = tf.cast(seq_length, tf.int32)
        return seq_length

    def make_cell():
      return tf.contrib.rnn.GRUCell(
          hidden_size, reuse=not is_training
      )

    cell = tf.contrib.rnn.MultiRNNCell(
        [make_cell() for _ in range(num_layers)])
    outputs, state = tf.nn.dynamic_rnn(cell, inputs, dtype=tf.float32, sequence_length=length(inputs))
    return outputs, state


class GaussianPolicyEstimator():
    """
    Policy Function approximator. Given a observation, returns probabilities
    over all possible actions.

    Args:
      num_outputs: Size of the action space.
      input_shape: List of input shape, batch size is leading dimension
      temporal_input_shape: List of temporal_input shape, batch size is leading dimension
      reuse: If true, an existing shared network will be re-used.
      trainable: If true we add train ops to the network.
        Actor threads that don't update their local models and don't need
        train ops would set this to false.
    """

    def __init__(self, num_actions, input_shape, temporal_input_shape, shared_layer, static_hidden_size=32, reuse=False, trainable=True):

        # assert input_shape[0] is None

        self.states = tf.placeholder(shape=input_shape, dtype=tf.float32, name="X")
        self.history = tf.placeholder(shape=temporal_input_shape, dtype=tf.float32, name="X_t")
        self.advantage = tf.placeholder(shape=(None, ), dtype=tf.float32, name='advantage')
        self.actions = tf.placeholder(shape=(None, num_actions), dtype=tf.float32, name="actions")

        X = tf.to_float(self.states)
        X_t = tf.to_float(self.history)

        # Graph shared with Value Net
        with tf.variable_scope("shared", reuse=reuse):
            output_t, state_t = shared_layer(X_t)
            dense_temporal = dense_layers(output_t[:, -1, :])
            dense_static = tf.contrib.layers.fully_connected(X, static_hidden_size, activation_fn=None)
            dense_output = tf.concat([dense_temporal, dense_static], axis=-1)

        with tf.variable_scope("policy_net"):
            normal_params = tf.contrib.layers.fully_connected(dense_output, num_actions * 2, activation_fn=None)
            normal_params = tf.reshape(normal_params, [-1, num_actions, 2])
            mu = normal_params[:, :, 0]
            sigma = tf.nn.softplus(normal_params[:, :, 1])

            self.predictions = {
                "mu": mu,
                "sigma": sigma,
            }

            self.entropy = 0.5 * (tf.log(2 * math.pi * tf.square(sigma) + keras.backend.epsilon()) + 1)
            self.entropy_mean = tf.reduce_mean(self.entropy, name="entropy_mean")

            nll = tf.log(sigma) + tf.square(self.actions - mu) / (2 * tf.square(sigma))
            loss = nll * self.advantage[:, None]
            self.loss = tf.reduce_mean(tf.reduce_sum(loss, name="loss"))

            tf.summary.scalar(self.loss.op.name, self.loss)
            tf.summary.scalar(self.entropy_mean.op.name, self.entropy_mean)
            tf.summary.histogram(self.entropy.op.name, self.entropy)

            if trainable:
                self.optimizer = tf.train.AdamOptimizer(1e-4)
                # self.optimizer = tf.train.RMSPropOptimizer(0.00025, 0.99, 0.0, 1e-6)
                self.grads_and_vars = self.optimizer.compute_gradients(self.loss)
                self.grads_and_vars = [[grad, var] for grad, var in self.grads_and_vars if grad is not None]
                self.train_op = self.optimizer.apply_gradients(
                    self.grads_and_vars,
                    global_step=tf.train.get_global_step()
                )

        # Merge summaries from this network and the shared network (but not the value net)
        # var_scope_name = tf.get_variable_scope().name
        # summary_ops = tf.get_collection(tf.GraphKeys.SUMMARIES)
        # sumaries = [s for s in summary_ops if var_scope_name in s.name]
        # self.summaries = tf.summary.merge(sumaries)


class ValueEstimator():
    """
    Policy Function approximator. Given a observation, returns probabilities
    over all possible actions.

    Args:
      num_outputs: Size of the action space.
      input_shape: List of input shape, batch size is leading dimension
      temporal_input_shape: List of temporal_input shape, batch size is leading dimension
      reuse: If true, an existing shared network will be re-used.
      trainable: If true we add train ops to the network.
        Actor threads that don't update their local models and don't need
        train ops would set this to false.
    """

    def __init__(self, num_actions, input_shape, temporal_input_shape, shared_layer, static_hidden_size=32, reuse=False, trainable=True):

        # assert input_shape[0] is None

        self.states = tf.placeholder(shape=input_shape, dtype=tf.float32, name="X")
        self.history = tf.placeholder(shape=temporal_input_shape, dtype=tf.float32, name="X_t")
        self.targets = tf.placeholder(shape=(None, ), dtype=tf.float32, name="targets")

        X = tf.to_float(self.states)
        X_t = tf.to_float(self.history)

        # Graph shared with Value Net
        with tf.variable_scope("shared", reuse=reuse):
            output_t, state_t = shared_layer(X_t)
            dense_temporal = dense_layers(output_t[:, -1, :])
            dense_static = tf.contrib.layers.fully_connected(X, static_hidden_size, activation_fn=None)
            dense_output = tf.concat([dense_temporal, dense_static], axis=-1)

        with tf.variable_scope("value_net"):
            self.logits = tf.contrib.layers.fully_connected(
                inputs=dense_output,
                num_outputs=1,
                activation_fn=None)
            self.logits = tf.squeeze(self.logits, squeeze_dims=[1], name="logits")

            self.losses = tf.squared_difference(self.logits, self.targets)
            self.loss = tf.reduce_sum(self.losses, name="loss")

            self.predictions = {
                "logits": self.logits
            }

            # Summaries
            prefix = tf.get_variable_scope().name
            tf.summary.scalar(self.loss.name, self.loss)
            tf.summary.scalar("{}/max_value".format(prefix), tf.reduce_max(self.logits))
            tf.summary.scalar("{}/min_value".format(prefix), tf.reduce_min(self.logits))
            tf.summary.scalar("{}/mean_value".format(prefix), tf.reduce_mean(self.logits))
            tf.summary.scalar("{}/reward_max".format(prefix), tf.reduce_max(self.targets))
            tf.summary.scalar("{}/reward_min".format(prefix), tf.reduce_min(self.targets))
            tf.summary.scalar("{}/reward_mean".format(prefix), tf.reduce_mean(self.targets))
            tf.summary.histogram("{}/reward_targets".format(prefix), self.targets)
            tf.summary.histogram("{}/values".format(prefix), self.logits)

            if trainable:
                self.optimizer = tf.train.AdamOptimizer(1e-4)
                # self.optimizer = tf.train.RMSPropOptimizer(0.00025, 0.99, 0.0, 1e-6)
                self.grads_and_vars = self.optimizer.compute_gradients(self.loss)
                self.grads_and_vars = [[grad, var] for grad, var in self.grads_and_vars if grad is not None]
                self.train_op = self.optimizer.apply_gradients(
                    self.grads_and_vars,
                    global_step=tf.train.get_global_step()
                )

        var_scope_name = tf.get_variable_scope().name
        summary_ops = tf.get_collection(tf.GraphKeys.SUMMARIES)
        sumaries = [s for s in summary_ops if var_scope_name in s.name]
        self.summaries = tf.summary.merge(sumaries)
