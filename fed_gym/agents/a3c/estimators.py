import math

import numpy as np

from tensorflow.contrib import keras
import tensorflow as tf


def build_shared_network(X, add_summaries=False):
    """
    Builds a 3-layer network conv -> conv -> fc as described
    in the A3C paper. This network is shared by both the policy and value net.

    Args:
      X: Inputs
      add_summaries: If true, add layer summaries to Tensorboard.

    Returns:
      Final layer activations.
    """

    # Fully connected layer
    fc1 = tf.contrib.layers.fully_connected(
        inputs=X,
        num_outputs=256,
        scope="fc_shared")

    if add_summaries:
        tf.contrib.layers.summarize_activation(fc1)

    return fc1

def dense_layers(X, add_summaries=False):
  fc1 = tf.contrib.layers.fully_connected(
    inputs=X,
    num_outputs=256,
    scope="fc1")

  if add_summaries:
    tf.contrib.layers.summarize_activation(fc1)

  return fc1


def true_length(sequence):
    used = tf.sign(tf.reduce_max(tf.abs(sequence), axis=2))
    seq_length = tf.reduce_sum(used, 1)
    seq_length = tf.cast(seq_length, tf.int32)
    return seq_length


def rnn_graph_lstm(inputs, hidden_size, num_layers, is_training):

    def make_cell():
      return tf.contrib.rnn.GRUCell(
          hidden_size, reuse=not is_training
      )

    cell = tf.contrib.rnn.MultiRNNCell(
        [make_cell() for _ in range(num_layers)])
    outputs, state = tf.nn.dynamic_rnn(cell, inputs, dtype=tf.float32, sequence_length=true_length(inputs))
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

    def __init__(self, num_actions, static_size, temporal_size, shared_layer, static_hidden_size=64, reuse=False, trainable=True, learning_rate=1e-4):

        self.states = tf.placeholder(shape=(None, static_size), dtype=tf.float32, name="X")
        self.history = tf.placeholder(shape=(None, None, temporal_size), dtype=tf.float32, name="X_t")
        self.advantages = tf.placeholder(shape=(None,), dtype=tf.float32, name='advantages')

        # Note: if actions are transformed, they should be provided in the untransformed shape (i.e N(mu, sig^2) space)
        self.actions = tf.placeholder(shape=(None, num_actions), dtype=tf.float32, name="actions")

        self.num_actions = num_actions
        self.static_size = static_size
        self.temporal_size = temporal_size

        X = tf.to_float(self.states)
        X_t = tf.to_float(self.history)

        with tf.variable_scope("shared", reuse=reuse):
            output_t, state_T = shared_layer(X_t)
            dense_temporal = dense_layers(state_T[-1])
            dense_static = tf.contrib.layers.fully_connected(X, static_hidden_size * 2)
            dense_static = tf.contrib.layers.fully_connected(dense_static, static_hidden_size)
            dense_output = tf.concat([dense_temporal, dense_static], axis=-1)

        with tf.variable_scope("policy_net"):

            normal_params = tf.contrib.layers.fully_connected(dense_output, static_hidden_size * 2)
            normal_params = tf.contrib.layers.fully_connected(normal_params, static_hidden_size)
            normal_params = 4. * tf.contrib.layers.fully_connected(normal_params, num_actions * 2, activation_fn=tf.nn.tanh)
            normal_params = tf.reshape(normal_params, [-1, num_actions, 2])
            mu = normal_params[:, :, 0]
            sigma = tf.nn.softplus(normal_params[:, :, 1] + keras.backend.epsilon())

            self.predictions = {
                "mu": mu,
                "sigma": sigma,
            }

            self.entropy = 0.5 * (tf.log(2 * math.pi * tf.square(sigma) + keras.backend.epsilon()) + 1)
            self.entropy_mean = tf.reduce_mean(self.entropy, name="entropy_mean")

            nll = tf.log(sigma + keras.backend.epsilon()) + tf.square(self.actions - mu) / (2 * tf.square(sigma))
            loss = nll * self.advantages[:, None]
            self.loss = tf.identity(tf.reduce_sum(loss) - 1e-3 * self.entropy_mean, name='loss')

            tf.summary.scalar(self.loss.op.name, self.loss)
            tf.summary.scalar(self.entropy_mean.op.name, self.entropy_mean)
            tf.summary.histogram('entropy', self.entropy)
            tf.summary.histogram('nll', nll)
            tf.summary.histogram('advantages', self.advantages)
            tf.summary.histogram('actions', self.actions)
            tf.summary.histogram('sigmoid_actions', tf.nn.sigmoid(self.actions))
            tf.summary.histogram('mu', mu)
            tf.summary.histogram('sigma', sigma)

            if trainable:
                self.optimizer = tf.train.AdamOptimizer(learning_rate)
                self.grads_and_vars = self.optimizer.compute_gradients(self.loss)
                self.grads_and_vars = [[grad, var] for grad, var in self.grads_and_vars if grad is not None]
                self.train_op = self.optimizer.apply_gradients(
                    self.grads_and_vars,
                    global_step=tf.train.get_global_step()
                )

        # Merge summaries from this network and the shared network (but not the value net)
        var_scope_name = tf.get_variable_scope().name
        summary_ops = tf.get_collection(tf.GraphKeys.SUMMARIES)
        sumaries = [s for s in summary_ops if var_scope_name in s.name]
        self.summaries = tf.summary.merge(sumaries)


class ValueEstimator():
    """
    Value Function approximator.

    Args:
      num_outputs: Size of the action space.
      input_shape: List of input shape, batch size is leading dimension
      temporal_input_shape: List of temporal_input shape, batch size is leading dimension
      reuse: If true, an existing shared network will be re-used.
      trainable: If true we add train ops to the network.
        Actor threads that don't update their local models and don't need
        train ops would set this to false.
    """

    def __init__(self, static_size, temporal_size, shared_layer, static_hidden_size=64, reuse=False, trainable=True, learning_rate=1e-3):

        self.static_size = static_size
        self.temporal_size = temporal_size

        self.states = tf.placeholder(shape=(None, static_size), dtype=tf.float32, name="X")
        self.seq_length = tf.placeholder(shape=(None,), dtype=tf.int32, name='seq_length')
        self.history = tf.placeholder(shape=(None, None, temporal_size), dtype=tf.float32, name="X_t")
        self.targets = tf.placeholder(shape=(None, ), dtype=tf.float32, name="targets")

        X = tf.to_float(self.states)
        X_t = tf.to_float(self.history)

        with tf.variable_scope("shared", reuse=reuse):
            output_t, state_T = shared_layer(X_t)
            dense_temporal = dense_layers(state_T[-1])
            dense_static = tf.contrib.layers.fully_connected(X, static_hidden_size * 2)
            dense_static = tf.contrib.layers.fully_connected(dense_static, static_hidden_size)
            dense_output = tf.concat([dense_temporal, dense_static], axis=-1)

        with tf.variable_scope("value_net"):

            self.logits = tf.contrib.layers.fully_connected(
                inputs=dense_output,
                num_outputs=1,
                activation_fn=tf.nn.softplus
            )
            self.logits = tf.squeeze(self.logits, squeeze_dims=[1], name="logits")

            self.losses = tf.squared_difference(self.logits, self.targets)
            self.loss = tf.reduce_sum(self.losses, name="loss")

            self.predictions = {
                "logits": self.logits
            }

            # Summaries
            prefix = tf.get_variable_scope().name
            tf.summary.scalar(self.loss.op.name, self.loss)
            tf.summary.scalar("{}/max_value".format(prefix), tf.reduce_max(self.logits))
            tf.summary.scalar("{}/min_value".format(prefix), tf.reduce_min(self.logits))
            tf.summary.scalar("{}/mean_value".format(prefix), tf.reduce_mean(self.logits))
            tf.summary.scalar("{}/reward_max".format(prefix), tf.reduce_max(self.targets))
            tf.summary.scalar("{}/reward_min".format(prefix), tf.reduce_min(self.targets))
            tf.summary.scalar("{}/reward_mean".format(prefix), tf.reduce_mean(self.targets))
            tf.summary.histogram("{}/capital".format(prefix), tf.exp(self.states[0, :]))
            tf.summary.histogram("{}/reward_targets".format(prefix), self.targets)
            tf.summary.histogram("{}/values".format(prefix), self.logits)

            if trainable:
                self.optimizer = tf.train.AdamOptimizer(learning_rate)
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
