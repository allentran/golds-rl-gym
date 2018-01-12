import itertools
import collections
import math

import numpy as np
import tensorflow as tf

from estimators import ValueEstimator, GaussianPolicyEstimator, DiscreteAndContPolicyEstimator

Transition = collections.namedtuple(
    "Transition",
    ["state", "action", "reward", "next_state", "done"]
)


def sigmoid(x):
    "Numerically-stable sigmoid function."
    if isinstance(x, float):
        if x >= 0:
            z = np.exp(-x)
            return 1 / (1 + z)
        else:
            z = np.exp(x)
            return z / (1 + z)
    else:
        pos_mask = x >= 0
        z = np.zeros_like(x)
        transformed = np.zeros_like(z)
        z[pos_mask] = np.exp(-x[pos_mask])
        z[~pos_mask] = np.exp(x[~pos_mask])
        transformed[pos_mask] = 1 / (1 + z[pos_mask])
        transformed[~pos_mask] = z[~pos_mask] / (1 + z[~pos_mask])
        return transformed


def make_copy_params_op(v1_list, v2_list):
    """
    Creates an operation that copies parameters from variable in v1_list to variables in v2_list.
    The ordering of the variables in the lists must be identical.
    """
    v1_list = list(sorted(v1_list, key=lambda v: v.name))
    v2_list = list(sorted(v2_list, key=lambda v: v.name))

    update_ops = []
    for v1, v2 in zip(v1_list, v2_list):
        op = v2.assign(v1)
        update_ops.append(op)

    return update_ops


def make_train_op(local_estimator, global_estimator):
    """
    Creates an op that applies local estimator gradients
    to the global estimator.
    """
    local_grads, _ = zip(*local_estimator.grads_and_vars)
    # Clip gradients
    local_grads, _ = tf.clip_by_global_norm(local_grads, 5.0)
    _, global_vars = zip(*global_estimator.grads_and_vars)
    local_global_grads_and_vars = list(zip(local_grads, global_vars))
    return global_estimator.optimizer.apply_gradients(
        local_global_grads_and_vars,
        global_step=tf.train.get_global_step()
    )


class Worker(object):
    """
    An A3C worker thread. Runs episodes locally and updates global shared value and policy nets.

    Args:
      name: A unique name for this worker
      env: The Gym environment used by this worker
      policy_net: Instance of the globally shared policy net
      value_net: Instance of the globally shared value net
      shared_layer: Shared layer between value/policy net
      global_counter: Iterator that holds the global step
      discount_factor: Reward discount factor
      summary_writer: A tf.train.SummaryWriter for Tensorboard summaries
      max_global_steps: If set, stop coordinator when global_counter > max_global_steps
    """
    def __init__(self, name, env, policy_net, value_net, shared_layer, global_counter, discount_factor=0.99, summary_writer=None, max_global_steps=None):
        self.name = name
        self.discount_factor = discount_factor
        self.max_global_steps = max_global_steps
        self.global_step = tf.train.get_global_step()
        self.global_policy_net = policy_net
        self.global_value_net = value_net
        self.global_counter = global_counter
        self.local_counter = itertools.count()
        self.summary_writer = summary_writer
        self.env = env

        # Create local policy/value nets that are not updated asynchronously
        with tf.variable_scope(name):
            self.policy_net = GaussianPolicyEstimator(
                policy_net.num_actions, static_size=policy_net.static_size, temporal_size=policy_net.temporal_size,
                shared_layer=shared_layer
            )
            self.value_net = ValueEstimator(
                static_size=policy_net.static_size, temporal_size=policy_net.temporal_size,
                shared_layer=shared_layer,
            )

        # Op to copy params from global policy/valuenets
        self.copy_params_op = make_copy_params_op(
            tf.contrib.slim.get_variables(scope="global", collection=tf.GraphKeys.TRAINABLE_VARIABLES),
            tf.contrib.slim.get_variables(scope=self.name+'/', collection=tf.GraphKeys.TRAINABLE_VARIABLES)
        )

        self.vnet_train_op = make_train_op(self.value_net, self.global_value_net)
        self.pnet_train_op = make_train_op(self.policy_net, self.global_policy_net)

        self.state = None
        self.history = []

    def run(self, sess, coord, t_max, always_bootstrap=False, max_seq_length=5):
        with sess.as_default(), sess.graph.as_default():
            # Initial state
            self.state = self.env.reset()
            self.history.append(self.process_state(self.state))

            try:
                while not coord.should_stop():
                    # Copy Parameters from the global networks
                    sess.run(self.copy_params_op)

                    # Collect some experience
                    transitions, local_t, global_t = self.run_n_steps(t_max, sess)

                    if self.max_global_steps is not None and global_t >= self.max_global_steps:
                        tf.logging.info("Reached global step {}. Stopping.".format(global_t))
                        coord.request_stop()
                        return

                    # Update the global networks
                    self.update(transitions, sess, always_bootstrap=always_bootstrap, max_seq_length=max_seq_length)
                    self.history = self.history[-(2 * t_max):]

            except tf.errors.CancelledError:
                return

    def _policy_net_predict(self, state, history, sess):
        feed_dict = {
            self.policy_net.states: [state],
            self.policy_net.history: [history],
        }
        preds = sess.run(self.policy_net.predictions, feed_dict)
        return preds["mu"], preds["sigma"]

    def _value_net_predict(self, state, history, sess):
        feed_dict = {
            self.value_net.states: [state],
            self.value_net.history: [history],
        }
        preds = sess.run(self.value_net.predictions, feed_dict)
        return preds["logits"][0]

    @staticmethod
    def get_temporal_states(history):
        raise NotImplementedError

    @staticmethod
    def get_random_action(mu, sigma, n_actions):
        raise NotImplementedError

    def run_n_steps(self, n, sess):
        transitions = []
        for _ in xrange(n):
            # Take a step
            action_mu, action_sigma = self._policy_net_predict(
                self.process_state(self.state), self.get_temporal_states(self.history), sess
            )
            action = self.get_random_action(action_mu, action_sigma, self.policy_net.num_actions)
            next_state, reward, done, _ = self.env.step(self.transform_raw_action(action))

            # Store transition
            transitions.append(Transition(
                state=self.process_state(self.state),
                action=action,
                reward=reward,
                next_state=self.process_state(next_state),
                done=done)
            )

            # Increase local and global counters
            local_t = next(self.local_counter)
            global_t = next(self.global_counter)

            if done:
                self.state = self.env.reset()
                self.history.append(self.process_state(self.state))
                break
            else:
                self.state = next_state
                self.history.append(self.process_state(next_state))

        return transitions, local_t, global_t

    def update(self, transitions, sess, always_bootstrap=False, max_seq_length=5):
        """
        Updates global policy and value networks based on collected experience

        Args:
          transitions: A list of experience transitions
          sess: A Tensorflow session
        """

        # If we episode was not done we bootstrap the value from the last state
        reward = 0.0
        history = self.history
        if not transitions[-1].done or always_bootstrap:
            state = transitions[-1].next_state
            reward = self._value_net_predict(state, self.get_temporal_states(history), sess)

        # Accumulate minibatch exmaples
        states = []
        policy_advantages = []
        value_targets = []
        actions = []
        temporal_states = []
        history = np.vstack(history)

        for idx, transition in enumerate(transitions[::-1]):
            reward = transition.reward + self.discount_factor * reward
            processed_state = transition.state
            history_t = history[:-(idx + 1)]
            policy_advantage = reward - self._value_net_predict(processed_state, self.get_temporal_states(history_t), sess)
            # Accumulate updates
            temporal_states.append(self.get_temporal_states(history_t))
            states.append(processed_state)
            actions.append(transition.action)
            policy_advantages.append(policy_advantage)
            value_targets.append(reward)

        temporal_state_matrix = tf.keras.preprocessing.sequence.pad_sequences(
            temporal_states, dtype='float32', padding='post', maxlen=max_seq_length
        )

        feed_dict = {
            self.policy_net.states: np.array(states),
            self.policy_net.history: temporal_state_matrix,
            self.policy_net.advantages: np.array(policy_advantages).flatten(),
            self.policy_net.actions: np.array(actions).reshape((-1, self.global_policy_net.num_actions)),
            self.value_net.states: np.array(states),
            self.value_net.history: temporal_state_matrix,
            self.value_net.targets: np.array(value_targets).flatten(),
        }

        # Train the global estimators using local gradients
        global_step, pnet_loss, vnet_loss, _, _, pnet_summaries, vnet_summaries = sess.run(
            [
                self.global_step,
                self.policy_net.loss,
                self.value_net.loss,
                self.pnet_train_op,
                self.vnet_train_op,
                self.policy_net.summaries,
                self.value_net.summaries
            ],
            feed_dict
        )

        # Write summaries
        if self.summary_writer is not None:
            self.summary_writer.add_summary(pnet_summaries, global_step)
            self.summary_writer.add_summary(vnet_summaries, global_step)
            self.summary_writer.flush()

        return pnet_loss, vnet_loss, pnet_summaries, vnet_summaries

    @staticmethod
    def transform_raw_action(*raw_actions):
        raise NotImplementedError

    @staticmethod
    def untransform_action(*transformed_actions):
        raise NotImplementedError

    @staticmethod
    def process_state(raw_state, **kwargs):
        return raw_state


class SolowWorker(Worker):

    @staticmethod
    def get_temporal_states(history):
        return np.array(history)

    @staticmethod
    def process_state(raw_state):
        return np.array([np.log(raw_state[0] / 100.), raw_state[1]]).flatten()

    @staticmethod
    def get_random_action(mu, sigma, n_actions):
        raw_action = np.random.normal(mu, sigma)[0]
        return raw_action

    @staticmethod
    def transform_raw_action(*raw_actions):
        return min(0.99, sigmoid(raw_actions[0]))


class TradeWorker(Worker):
    @staticmethod
    def process_state(raw_state):
        new_states = []
        n_assets = len(raw_state) - 1

        new_states.append(np.log(raw_state[0] + 1e-4))

        quantity = raw_state[1:1 + n_assets]
        new_states += np.log(quantity + 1).tolist()
        prices = raw_state[1 + n_assets:]
        new_states += np.log(prices).tolist()

        return np.array(new_states).flatten()

    @staticmethod
    def get_temporal_states(history):
        return np.vstack(history)

    @staticmethod
    def get_random_action(mu, sigma, n_actions):
        raw_action = (mu + sigma * np.random.normal(size=(n_actions, ))).flatten()
        return TradeWorker.transform_raw_action(raw_action)

    @staticmethod
    def transform_raw_action(*raw_actions):
        return np.tanh(raw_actions[0])

    @staticmethod
    def untransform_action(*transformed_actions):
        return np.arctanh(transformed_actions[0])


class TickerGatedTraderWorker(Worker):

    def __init__(self, name, env, policy_net, value_net, shared_layer, global_counter, discount_factor=0.99, summary_writer=None, max_global_steps=None):
        self.name = name
        self.discount_factor = discount_factor
        self.max_global_steps = max_global_steps
        self.global_step = tf.train.get_global_step()
        self.global_policy_net = policy_net
        self.global_value_net = value_net
        self.global_counter = global_counter
        self.local_counter = itertools.count()
        self.summary_writer = summary_writer
        self.env = env
        self.n_assets = policy_net.num_assets

        # Create local policy/value nets that are not updated asynchronously
        with tf.variable_scope(name):
            self.policy_net = DiscreteAndContPolicyEstimator(
                policy_net.num_assets, static_size=policy_net.static_size, temporal_size=policy_net.temporal_size,
                shared_layer=shared_layer
            )
            self.value_net = ValueEstimator(
                static_size=policy_net.static_size, temporal_size=policy_net.temporal_size,
                shared_layer=shared_layer,
            )

        # Op to copy params from global policy/valuenets
        self.copy_params_op = make_copy_params_op(
            tf.contrib.slim.get_variables(scope="global", collection=tf.GraphKeys.TRAINABLE_VARIABLES),
            tf.contrib.slim.get_variables(scope=self.name+'/', collection=tf.GraphKeys.TRAINABLE_VARIABLES)
        )

        self.vnet_train_op = make_train_op(self.value_net, self.global_value_net)
        self.pnet_train_op = make_train_op(self.policy_net, self.global_policy_net)

        self.state = None
        self.history = []

    @staticmethod
    def process_state(raw_state, **kwargs):
        cash = raw_state[0]
        n_assets = kwargs['n_assets']
        quantities = raw_state[1: 1 + n_assets]
        prices = raw_state[1 + n_assets: -n_assets]
        volumes = raw_state[-n_assets:]

        state = [np.log(cash + 1e-4)]
        for idx in xrange(n_assets):
            state.append(np.log(quantities[idx] + 1))
        for idx in xrange(n_assets):
            state.append(np.log(prices[idx]))
        for idx in xrange(n_assets):
            state.append(volumes[idx])
        return np.array(state).flatten()

    @staticmethod
    def get_temporal_states(history, **kwargs):
        n_assets = kwargs['n_assets']
        return np.vstack(history)[:, 1 + n_assets:]

    @staticmethod
    def transform_raw_action(*actions):
        return sigmoid(actions[0])

    @staticmethod
    def untransform_action(*actions):
        return np.log(actions[0] / (1 - actions[0]))

    @staticmethod
    def get_random_discrete_action(probs):
        cum_probs = probs.cumsum(axis=1)
        u = np.random.rand(len(cum_probs), 1)
        return (u < cum_probs).argmax(axis=1)

    @staticmethod
    def get_random_action(mu, sigma, choices):
        row_idx = np.arange(len(choices))
        mu = mu[row_idx, choices]
        sigma = sigma[row_idx, choices]
        return mu + sigma * np.random.normal(size=mu.shape)

    def run_n_steps(self, n, sess):
        transitions = []
        for _ in xrange(n):
            # Take a step
            action_mu, action_sigma, discrete_probs = self._policy_net_predict(
                self.process_state(self.state, n_assets=self.n_assets),
                self.get_temporal_states(self.history, n_assets=self.n_assets),
                sess
            )
            discrete_choices = self.get_random_discrete_action(discrete_probs[0])
            cont_action = self.get_random_action(action_mu[0], action_sigma[0], discrete_choices)
            next_state, reward, done, _ = self.env.step([discrete_choices, self.transform_raw_action(cont_action)])

            # Store transition
            transitions.append(Transition(
                state=self.process_state(self.state, n_assets=self.n_assets),
                action=[discrete_choices, cont_action],
                reward=reward,
                next_state=self.process_state(next_state, n_assets=self.n_assets),
                done=done)
            )

            # Increase local and global counters
            local_t = next(self.local_counter)
            global_t = next(self.global_counter)

            if done:
                self.state = self.env.reset()
                self.history.append(self.process_state(self.state, n_assets=self.n_assets))
                break
            else:
                self.state = next_state
                self.history.append(self.process_state(next_state, n_assets=self.n_assets))

        return transitions, local_t, global_t

    def _policy_net_predict(self, state, history, sess):
        feed_dict = {
            self.policy_net.states: [state],
            self.policy_net.history: [history],
        }
        preds = sess.run(self.policy_net.predictions, feed_dict)
        return preds["mu"], preds["sigma"], preds['probs']

    def update(self, transitions, sess, always_bootstrap=False, max_seq_length=5):
        """
        Updates global policy and value networks based on collected experience

        Args:
          transitions: A list of experience transitions
          sess: A Tensorflow session
        """

        # If we episode was not done we bootstrap the value from the last state
        reward = 0.0
        history = self.history
        if not transitions[-1].done or always_bootstrap:
            state = transitions[-1].next_state
            reward = self._value_net_predict(
                state, self.get_temporal_states(history, n_assets=self.n_assets), sess
            )

        # Accumulate minibatch exmaples
        states = []
        policy_advantages = []
        value_targets = []
        transformed_cont_actions = []
        discrete_actions = []
        temporal_states = []
        history = np.vstack(history)

        for idx, transition in enumerate(transitions[::-1]):
            reward = transition.reward + self.discount_factor * reward
            history_t = history[:-(idx + 1)]
            policy_advantage = reward - self._value_net_predict(
                transition.state, self.get_temporal_states(history_t, n_assets=self.n_assets), sess
            )
            # Accumulate updates
            temporal_states.append(self.get_temporal_states(history_t, n_assets=self.n_assets))
            states.append(transition.state)
            discrete_actions.append(transition.action[0])
            transformed_cont_actions.append(transition.action[1][None, :])
            policy_advantages.append(policy_advantage)
            value_targets.append(reward)

        temporal_state_matrix = tf.keras.preprocessing.sequence.pad_sequences(
            temporal_states, dtype='float32', padding='post', maxlen=max_seq_length
        )

        feed_dict = {
            self.policy_net.states: np.array(states),
            self.policy_net.history: temporal_state_matrix,
            self.policy_net.advantages: policy_advantages,
            self.policy_net.discrete_actions: discrete_actions,
            self.policy_net.actions: np.vstack(transformed_cont_actions),
            self.value_net.states: np.array(states),
            self.value_net.history: temporal_state_matrix,
            self.value_net.targets: value_targets,
        }

        # Train the global estimators using local gradients
        global_step, pnet_loss, vnet_loss, _, _, pnet_summaries, vnet_summaries = sess.run(
            [
                self.global_step,
                self.policy_net.loss,
                self.value_net.loss,
                self.pnet_train_op,
                self.vnet_train_op,
                self.policy_net.summaries,
                self.value_net.summaries
            ],
            feed_dict
        )

        # Write summaries
        if self.summary_writer is not None:
            self.summary_writer.add_summary(pnet_summaries, global_step)
            self.summary_writer.add_summary(vnet_summaries, global_step)
            self.summary_writer.flush()

        return pnet_loss, vnet_loss, pnet_summaries, vnet_summaries
