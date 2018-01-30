import math

import numpy as np

from tensorflow.contrib import keras
import tensorflow as tf


def true_length(sequence):
    used = tf.sign(tf.reduce_max(tf.abs(sequence), axis=2))
    seq_length = tf.reduce_sum(used, 1)
    seq_length = tf.cast(seq_length, tf.int32)
    return seq_length


def rnn_graph_lstm(temporal_inputs, static_inputs, hidden_size, num_layers, is_training):

    def make_cell():
      return tf.nn.rnn_cell.GRUCell(
          hidden_size, reuse=not is_training
      )

    cell = tf.nn.rnn_cell.MultiRNNCell(
        [make_cell() for _ in range(num_layers)])
    outputs, state = tf.nn.dynamic_rnn(cell, temporal_inputs, dtype=tf.float32, sequence_length=true_length(temporal_inputs))
    rnn_last = state[-1]

    dense_temporal = tf.layers.dense(rnn_last, hidden_size * 2, activation=tf.nn.relu)
    dense_static = tf.layers.dense(static_inputs, hidden_size * 2, activation=tf.nn.relu)
    dense_static = tf.layers.dense(dense_static, hidden_size, activation=tf.nn.relu)
    return tf.concat([dense_temporal, dense_static], axis=-1)


class StateProcessor(object):
    def __init__(self, scales):
        self.scales = scales

    def process_temporal_states(self, history):
        raise NotImplementedError

    def process_state(self, state):
        return state / self.scales


class TickerTraderStateProcessor(StateProcessor):
    def __init__(self, n_assets):
        super(TickerTraderStateProcessor, self).__init__(None)
        self.n_assets = n_assets

    def process_state(self, raw_state):
        cash = raw_state[0]
        quantities = raw_state[1: 1 + self.n_assets]
        prices = raw_state[1 + self.n_assets: -self.n_assets]
        volumes = raw_state[-self.n_assets:]

        state = [np.log(cash + 1e-4)]
        for idx in range(self.n_assets):
            state.append(np.log(quantities[idx] + 1))
        for idx in range(self.n_assets):
            state.append(np.log(prices[idx]))
        for idx in range(self.n_assets):
            state.append(volumes[idx])
        return np.array(state).flatten()

    def process_temporal_states(self, history):
        return np.vstack(history)[:, 1 + self.n_assets:]


class SolowStateProcessor(StateProcessor):
    def __init__(self):
        super(SolowStateProcessor, self).__init__(np.array([100., 1.]))

    def process_temporal_states(self, history):
        if len(history) == 1:
            return np.array(history[0]).reshape((1, -1))
        return np.array(history)


class PolicyEstimator(object):
    def predict(self, state, history, sess, batch=False):
        feed_dict = {
            self.states: [state] if not batch else state,
            self.history: [history] if not batch else history,
        }
        return sess.run(self.predictions, feed_dict)


class DiscreteAndContPolicyEstimator(PolicyEstimator):
    """
    Policy Function approximator. Given a observation, returns probabilities
    over all possible actions.

    Args:
      num_discrete: Size of the action space.
      input_shape: List of input shape, batch size is leading dimension
      temporal_input_shape: List of temporal_input shape, batch size is leading dimension
      reuse: If true, an existing shared network will be re-used.
      trainable: If true we add train ops to the network.
        Actor threads that don't update their local models and don't need
        train ops would set this to false.
    """
    num_actions = 3
    BUY_IDX = 1
    SELL_IDX = 2

    def __init__(self, num_assets, static_size, temporal_size, shared_layer, static_hidden_size=128, trainable=True, learning_rate=1e-4, seed=None, reuse=False):
        self.states = tf.placeholder(shape=(None, static_size), dtype=tf.float32, name="X")
        self.history = tf.placeholder(shape=(None, None, temporal_size), dtype=tf.float32, name="X_t")
        self.advantages = tf.placeholder(shape=(None,), dtype=tf.float32, name='advantages')
        self.batch_size = tf.placeholder(tf.int32, name='batch_size')

        self.discrete_actions = tf.placeholder(shape=(None, num_assets), dtype=tf.int32, name="discrete_actions")
        # Note: if actions are transformed, they should be provided in the untransformed shape (i.e N(mu, sig^2) space)
        self.actions = tf.placeholder(shape=(None, num_assets), dtype=tf.float32, name="cont_actions")

        self.num_assets = num_assets
        self.static_size = static_size
        self.temporal_size = temporal_size

        X = tf.to_float(self.states)
        X_t = tf.to_float(self.history)

        if seed:
            tf.set_random_seed(seed)

        with tf.variable_scope("shared", reuse=reuse):
            dense_output = shared_layer(X_t, X)

        with tf.variable_scope("policy_net"):
            class_hidden = tf.layers.dense(dense_output, static_hidden_size * 2, activation=tf.nn.relu)
            class_hidden = tf.layers.dense(class_hidden, static_hidden_size, activation=tf.nn.relu)
            discrete_logits = tf.layers.dense(
                class_hidden, num_assets * self.num_actions, activation=None
            )
            discrete_probs = tf.nn.softmax(tf.reshape(discrete_logits, (-1, num_assets, self.num_actions)))

            normal_params = tf.layers.dense(dense_output, static_hidden_size * 2, activation=tf.nn.relu)
            normal_params = tf.layers.dense(normal_params, static_hidden_size, activation=tf.nn.relu)
            normal_params = tf.layers.dense(
                normal_params, num_assets * self.num_actions * 2, activation=None
            )
            normal_params = tf.reshape(normal_params, [-1, self.num_assets, self.num_actions, 2])
            mu = normal_params[:, :, :, 0]
            sigma = tf.nn.softplus(normal_params[:, :, :, 1]) + keras.backend.epsilon()

            one_hot_actions = tf.one_hot(self.discrete_actions, depth=self.num_actions, dtype=tf.float32)

            action_probs = tf.reduce_sum(one_hot_actions * discrete_probs, axis=-1)
            mu_action = tf.reduce_sum(one_hot_actions * mu, axis=-1)
            sig_action = tf.reduce_sum(one_hot_actions * sigma, axis=-1)

            cont_dist = tf.distributions.Normal(mu_action, sig_action)

            self.predictions = {
                "mu": mu,
                "sigma": sigma,
                "probs": discrete_probs,
            }

            discrete_entropy = - tf.reduce_sum(discrete_probs * tf.log(discrete_probs), axis=-1)
            cont_entropy = cont_dist.entropy()
            self.entropy_mean = tf.reduce_mean(cont_entropy + discrete_entropy, name="entropy_mean")

            nll_discrete = - tf.log(action_probs)
            nll_cont = - cont_dist.log_prob(self.actions)
            loss = (nll_discrete + nll_cont) * self.advantages[:, None]

            self.loss = tf.identity(tf.reduce_sum(loss) - 0 * self.entropy_mean, name='loss')

            tf.summary.scalar(self.loss.op.name, self.loss)
            tf.summary.scalar(self.entropy_mean.op.name, self.entropy_mean)
            tf.summary.histogram('cont_entropy', cont_entropy)
            tf.summary.histogram('discrete_entropy', discrete_entropy)
            tf.summary.histogram('nll', nll_cont + nll_discrete)
            tf.summary.histogram('advantages', self.advantages)
            tf.summary.histogram('buy_prob', discrete_probs[:, :, self.BUY_IDX])
            tf.summary.histogram('sell_prob', discrete_probs[:, :, self.SELL_IDX])
            tf.summary.histogram('z_var', (self.actions - mu_action) / sig_action)
            tf.summary.histogram('sigmoid_actions', tf.nn.sigmoid(self.actions))
            tf.summary.histogram('tanh_actions', tf.nn.tanh(self.actions))
            tf.summary.histogram('mu', mu_action)
            tf.summary.histogram('sigma', sig_action)

            if trainable:
                learning_rate = tf.train.exponential_decay(
                    learning_rate, tf.train.get_global_step(), 100000, 0.96, staircase=False
                )
                self.optimizer = tf.train.RMSPropOptimizer(learning_rate, epsilon=0.1, decay=0.99)
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


class DiscretePolicyEstimator(PolicyEstimator):
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

    def __init__(self, num_outputs, num_choices, static_size, temporal_size, shared_layer, static_hidden_size=128, reuse=False, trainable=True, learning_rate=1e-4, seed=None, lb=-5., ub=5.):

        self.states = tf.placeholder(shape=(None, static_size), dtype=tf.float32, name="X")
        self.history = tf.placeholder(shape=(None, None, temporal_size), dtype=tf.float32, name="X_t")
        self.advantages = tf.placeholder(shape=(None,), dtype=tf.float32, name='advantages')

        # Note: if actions are transformed, they should be provided in the untransformed shape (i.e N(mu, sig^2) space)
        self.actions = tf.placeholder(shape=(None, num_outputs), dtype=tf.int32, name="actions")

        self.num_outputs = num_outputs
        self.num_choices = num_choices
        self.static_size = static_size
        self.temporal_size = temporal_size

        X = tf.to_float(self.states)
        X_t = tf.to_float(self.history)

        if seed:
            tf.set_random_seed(seed)

        with tf.variable_scope("shared", reuse=reuse):
            dense_output = shared_layer(X_t, X)

        with tf.variable_scope("policy_net"):

            with tf.variable_scope('probs'):
                y = tf.layers.dense(dense_output, static_hidden_size * 2, activation=tf.nn.relu)
                y = tf.layers.dense(y, static_hidden_size, activation=tf.nn.relu)
                y = tf.layers.dense(y, num_outputs * num_choices, activation=None)
                y = tf.reshape(y, (-1, num_outputs, num_choices))
                probs = tf.nn.softmax(y)

            self.predictions = {
                "probs": probs,
            }

            self.entropy = - tf.reduce_mean(tf.reduce_sum(probs * tf.log(probs + tf.keras.backend.epsilon()), axis=-1), axis=-1)
            self.entropy_mean = tf.reduce_mean(self.entropy, name="entropy_mean")

            action_probs = tf.reduce_sum(tf.one_hot(self.actions, self.num_choices) * probs, axis=-1)
            nll = - tf.log(action_probs + tf.keras.backend.epsilon())
            loss = nll * self.advantages[:, None]
            self.loss = tf.identity(tf.reduce_sum(loss), name='loss')

            tf.summary.scalar(self.loss.op.name, self.loss)
            tf.summary.scalar(self.entropy_mean.op.name, self.entropy_mean)
            tf.summary.histogram('entropy', self.entropy)
            tf.summary.histogram('nll', nll)
            tf.summary.histogram('advantages', self.advantages)
            tf.summary.histogram('chosen_probs', action_probs)
            tf.summary.histogram('probs', probs)

            if trainable:
                learning_rate = tf.train.exponential_decay(
                    learning_rate, tf.train.get_global_step(), 100000, 0.96, staircase=False
                )
                self.optimizer = tf.train.RMSPropOptimizer(learning_rate, epsilon=0.1, decay=0.99)
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


class GaussianPolicyEstimator(PolicyEstimator):
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

    def __init__(self, num_actions, static_size, temporal_size, shared_layer, static_hidden_size=128, reuse=False, trainable=True, learning_rate=1e-4, seed=None, lb=-5., ub=5.):

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

        if seed:
            tf.set_random_seed(seed)

        with tf.variable_scope("shared", reuse=reuse):
            dense_output = shared_layer(X_t, X)

        with tf.variable_scope("policy_net"):

            with tf.variable_scope('mu'):
                mu = tf.layers.dense(dense_output, static_hidden_size * 2, activation=tf.nn.relu)
                mu = tf.layers.dense(mu, static_hidden_size, activation=tf.nn.tanh)
                mu = ((ub - lb) / 2.) * tf.layers.dense(mu, num_actions, activation=tf.nn.tanh) + ((lb + ub) / 2.)

            with tf.variable_scope('sigma', reuse=False):
                sigma = tf.layers.dense(dense_output, static_hidden_size * 2, activation=tf.nn.relu)
                sigma = tf.layers.dense(sigma, static_hidden_size, activation=tf.nn.tanh)
                sigma = tf.layers.dense(
                    sigma, num_actions, activation=tf.nn.sigmoid, bias_initializer=tf.constant_initializer(-1.)
                ) + 1e-3

            dist = tf.distributions.Normal(loc=mu, scale=sigma)

            self.predictions = {
                "mu": mu,
                "sigma": sigma,
            }

            self.entropy = dist.entropy()
            self.entropy_mean = tf.reduce_mean(self.entropy, name="entropy_mean")

            nll = - dist.log_prob(self.actions)
            loss = nll * self.advantages[:, None]
            self.loss = tf.identity(tf.reduce_sum(loss), name='loss')

            tf.summary.scalar(self.loss.op.name, self.loss)
            tf.summary.scalar(self.entropy_mean.op.name, self.entropy_mean)
            tf.summary.histogram('z_var', (self.actions - mu) / sigma)
            tf.summary.histogram('entropy', self.entropy)
            tf.summary.histogram('nll', nll)
            tf.summary.histogram('advantages', self.advantages)
            tf.summary.histogram('actions', self.actions)
            tf.summary.histogram('sigmoid_actions', tf.nn.sigmoid(self.actions))
            tf.summary.histogram('tanh_actions', tf.nn.tanh(self.actions))
            tf.summary.histogram('mu', mu)
            tf.summary.histogram('sigma', sigma)

            if trainable:
                learning_rate = tf.train.exponential_decay(
                    learning_rate, tf.train.get_global_step(), 100000, 0.96, staircase=False
                )
                self.optimizer = tf.train.RMSPropOptimizer(learning_rate, epsilon=0.1, decay=0.99)
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

    def __init__(self, static_size, temporal_size, shared_layer, static_hidden_size=128, reuse=False, trainable=True, learning_rate=1e-4, num_actions=2, scale=1.):

        self.static_size = static_size
        self.temporal_size = temporal_size

        self.states = tf.placeholder(shape=(None, static_size), dtype=tf.float32, name="X")
        self.seq_length = tf.placeholder(shape=(None,), dtype=tf.int32, name='seq_length')
        self.history = tf.placeholder(shape=(None, None, temporal_size), dtype=tf.float32, name="X_t")
        self.targets = tf.placeholder(shape=(None, ), dtype=tf.float32, name="targets")

        X = tf.to_float(self.states)
        X_t = tf.to_float(self.history)

        with tf.variable_scope("shared", reuse=reuse):
            dense_output = shared_layer(X_t, X)

        with tf.variable_scope("value_net"):
            dense_output = tf.layers.dense(dense_output, static_hidden_size * 2, activation=tf.nn.tanh)
            self.logits = scale * tf.layers.dense(
                inputs=dense_output,
                units=1,
                activation=None,
            )
            self.logits = tf.squeeze(self.logits, squeeze_dims=[1], name="logits")

            self.losses = tf.squared_difference(self.logits, self.targets)
            self.loss = 0.5 * tf.reduce_sum(self.losses / scale, name="loss")

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
            tf.summary.histogram("{}/losses".format(prefix), self.losses)
            tf.summary.histogram("{}/capital".format(prefix), tf.exp(self.states[:, 0]) - 1)
            if static_size == 2 * num_actions + 1:
                for idx in range(num_actions):
                    tf.summary.histogram("{}/quantity_{}".format(prefix, idx), tf.exp(self.states[:, 1 + idx]) - 1)
                    tf.summary.histogram("{}/prices_{}".format(prefix, idx), tf.exp(self.states[:, 1 + num_actions + idx]))
            tf.summary.histogram("{}/reward_targets".format(prefix), self.targets)
            tf.summary.histogram("{}/values".format(prefix), self.logits)

            if trainable:
                learning_rate = tf.train.exponential_decay(
                    learning_rate, tf.train.get_global_step(), 100000, 0.96, staircase=False
                )
                self.optimizer = tf.train.RMSPropOptimizer(learning_rate, epsilon=0.1, decay=0.99)
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
