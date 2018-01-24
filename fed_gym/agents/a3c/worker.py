import itertools
import collections
import math

import numpy as np
import tensorflow as tf

from .estimators import ValueEstimator, GaussianPolicyEstimator, DiscreteAndContPolicyEstimator, DiscretePolicyEstimator

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


class GaussianWorker(object):
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
    def __init__(self, name, env, policy_net, value_net, shared_layer, global_counter, discount_factor=0.99, summary_writer=None, max_global_steps=None, scale=1.):
        self.name = name
        self.discount_factor = discount_factor
        self._lambda = 0.9
        self.max_global_steps = max_global_steps
        self.global_step = tf.train.get_global_step()
        self.global_policy_net = policy_net
        self.global_value_net = value_net
        self.global_counter = global_counter
        self.local_counter = itertools.count()
        self.summary_writer = summary_writer
        self.env = env
        self.scale = scale

        # Create local policy/value nets that are not updated asynchronously
        with tf.variable_scope(name):
            self.policy_net = self.build_local_policy_net(policy_net, shared_layer)
            self.value_net = self.build_local_value_net(policy_net, shared_layer)

        # Op to copy params from global policy/valuenets
        self.copy_params_op = make_copy_params_op(
            tf.contrib.slim.get_variables(scope="global", collection=tf.GraphKeys.TRAINABLE_VARIABLES),
            tf.contrib.slim.get_variables(scope=self.name+'/', collection=tf.GraphKeys.TRAINABLE_VARIABLES)
        )

        self.vnet_train_op = make_train_op(self.value_net, self.global_value_net)
        self.pnet_train_op = make_train_op(self.policy_net, self.global_policy_net)

        self.state = None
        self.history = []
        self.debug = None

    def build_local_policy_net(self, global_policy_net, shared_layer):
        raise NotImplementedError

    def build_local_value_net(self, global_policy_net, shared_layer):
        return ValueEstimator(
            static_size=global_policy_net.static_size, temporal_size=global_policy_net.temporal_size,
            shared_layer=shared_layer,
            scale=self.scale,
            reuse=True,
            learning_rate=1e-4
        )

    def run(self, sess, coord, t_max, always_bootstrap=False, max_seq_length=3):
        with sess.as_default(), sess.graph.as_default():
            # Initial state
            self.state = self.env.reset()
            self.history.append(self.process_state(self.state))

            try:
                while not coord.should_stop():
                    # Copy Parameters from the global networks
                    sess.run(self.copy_params_op)

                    # Collect some experience
                    transitions, local_t, global_t, mus, done = self.run_n_steps(t_max, sess, max_seq_length=max_seq_length)

                    if self.max_global_steps is not None and global_t >= self.max_global_steps:
                        tf.logging.info("Reached global step {}. Stopping.".format(global_t))
                        coord.request_stop()
                        return

                    # Update the global networks
                    self.update(
                        transitions, sess, always_bootstrap=always_bootstrap, max_seq_length=max_seq_length
                    )
                    if done:
                        self.state = self.env.reset()
                        self.history = [self.process_state(self.state)]
                    else:
                        self.history = self.history[-(2 * t_max):]

            except tf.errors.CancelledError:
                return

    def _policy_net_predict(self, state, history, sess, batch=False):
        feed_dict = {
            self.policy_net.states: [state] if not batch else state,
            self.policy_net.history: [history] if not batch else history,
        }
        return sess.run(self.policy_net.predictions, feed_dict)

    def _value_net_predict_many(self, states, history, sess):
        feed_dict = {
            self.value_net.states: states,
            self.value_net.history: history,
        }
        preds = sess.run(self.value_net.predictions, feed_dict)
        return preds["logits"]

    def _value_net_predict(self, state, history, sess):
        feed_dict = {
            self.value_net.states: [state],
            self.value_net.history: [history],
        }
        preds = sess.run(self.value_net.predictions, feed_dict)
        return preds["logits"][0]

    def get_temporal_states(self, history):
        raise NotImplementedError

    @staticmethod
    def get_random_action(mu, sigma, n_actions):
        raise NotImplementedError

    def get_action_from_policy(self, processed_state, history, session):
        preds = self._policy_net_predict(
            processed_state, self.get_temporal_states(history), session
        )
        self.debug.append(preds['mu'][0][0])
        return self.get_random_action(preds['mu'][0][0], preds['sigma'][0][0], self.policy_net.num_actions)

    def run_n_steps(self, n, sess, max_seq_length=5):
        transitions = []
        self.debug = []
        for _ in range(n):
            # Take a step
            processed_state = self.process_state(self.state)
            action = self.get_action_from_policy(processed_state, self.history[-max_seq_length:], sess)
            next_state, reward, done, _ = self.env.step(self.transform_raw_action(*action))
            processed_next_state = self.process_state(next_state)
            # Store transition
            transitions.append(Transition(
                state=processed_state,
                action=action,
                reward=reward,
                next_state=processed_next_state,
                done=done)
            )

            # Increase local and global counters
            local_t = next(self.local_counter)
            global_t = next(self.global_counter)

            self.history.append(self.process_state(next_state))
            if not done:
                self.state = next_state
            else:
                self.state = None
                break

        return transitions, local_t, global_t, self.debug, done

    @staticmethod
    def get_random_discrete_action(probs):
        cum_probs = probs.cumsum(axis=1)
        u = np.random.rand(len(cum_probs), 1)
        return (u < cum_probs).argmax(axis=1)

    def update(self, transitions, sess, always_bootstrap=False, max_seq_length=5, done_penalty=0.):
        """
        Updates global policy and value networks based on collected experience

        Args:
          transitions: A list of experience transitions
          sess: A Tensorflow session
        """

        # If we episode was not done we bootstrap the value from the last state
        reward = done_penalty
        if not transitions[-1].done or always_bootstrap:
            reward = self._value_net_predict(
                transitions[-1].next_state,
                self.get_temporal_states(self.history[-max_seq_length:]),
                sess
            )

        steps = len(transitions)

        # Accumulate minibatch exmaples
        states = []
        advantages = []
        value_targets = []
        actions = []
        rewards = []
        temporal_states = []

        for t, transition in enumerate(transitions):
            rewards.append(transition.reward)
            actions.append(transition.action)
            processed_state = transition.state
            history_t = self.history[- (steps - t + max_seq_length): - (steps - t)]
            temporal_states.append(self.get_temporal_states(history_t))
            states.append(processed_state)

        temporal_state_matrix = tf.keras.preprocessing.sequence.pad_sequences(
            temporal_states, dtype='float32', padding='post', maxlen=max_seq_length
        )

        # V_s(t) has all V(s_t) for t=0, ..., T, T + 1
        V_st = self._value_net_predict_many(states, temporal_state_matrix, sess)
        V_st = np.concatenate([V_st, [reward]])
        temporal_state_matrix = np.flip(temporal_state_matrix, 0)

        # GAE deltas
        delta_ts = []
        for t in range(len(transitions)):
            delta_ts.append(
                rewards[t] + self.discount_factor * V_st[t + 1] - V_st[t]
            )
        delta_ts = np.array(delta_ts).flatten()

        # GAE advantages + TD(lambda)
        for t in range(len(transitions)):
            deltas = delta_ts[t:]
            gammas = self.discount_factor ** np.arange(len(deltas))
            a_n = (gammas * deltas).cumsum()
            weights = self._lambda ** np.arange(len(deltas))
            weights[:-1] *= 1 - self._lambda
            np.testing.assert_almost_equal(weights.sum(), 1., decimal=4)
            advantage = (a_n * weights).sum()
            advantages.append(advantage)
            value_targets.append(advantage + V_st[t])

        assert len(V_st) == len(transitions) + 1

        states = np.array(states[::-1])
        actions = actions[::-1]
        value_targets = np.array(value_targets[::-1])

        feed_dict = self.fill_feed_dict_for_update(states, temporal_state_matrix, advantages, actions, value_targets)

        # Train the global estimators using local gradients
        predictions, global_step, pnet_loss, vnet_loss, _, _, pnet_summaries, vnet_summaries = sess.run(
            [
                self.policy_net.predictions,
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

        return pnet_loss, vnet_loss, pnet_summaries, vnet_summaries, predictions

    def fill_feed_dict_for_update(self, states, temporal_state_matrix, advantages, actions, value_targets):
        actions = np.array(actions)
        return {
            self.policy_net.states: states,
            self.policy_net.history: temporal_state_matrix,
            self.policy_net.advantages: np.array(advantages[::-1]).flatten() / self.scale,
            self.policy_net.actions: actions.reshape((-1, self.global_policy_net.num_actions)),
            self.value_net.states: states,
            self.value_net.history: temporal_state_matrix,
            self.value_net.targets: value_targets.flatten(),
        }

    @staticmethod
    def transform_raw_action(*raw_actions):
        raise NotImplementedError

    def process_state(self, raw_state):
        return raw_state


class GridWorker(GaussianWorker):

    def __init__(self, name, env, policy_net, value_net, shared_layer, global_counter, discount_factor=0.99,
                 summary_writer=None, max_global_steps=None, scale=1.):
        super(GridWorker, self).__init__(name, env, policy_net, value_net, shared_layer, global_counter,
                                         discount_factor, summary_writer, max_global_steps, scale)

    def build_local_policy_net(self, global_policy_net, shared_layer):
        return DiscretePolicyEstimator(
            global_policy_net.num_outputs,
            global_policy_net.num_choices,
            static_size=global_policy_net.static_size,
            temporal_size=global_policy_net.temporal_size,
            shared_layer=shared_layer,
        )

    def get_action_from_policy(self, processed_state, history, session):
        preds = self._policy_net_predict(
            processed_state, self.get_temporal_states(history), session
        )
        self.debug.append(preds['probs'][0])
        return self.get_random_action(preds['probs'][0], None, None)

    @staticmethod
    def get_random_action(probs, sigma, n_actions):
        return [GaussianWorker.get_random_discrete_action(probs)]


class SolowWorker(GaussianWorker):

    def __init__(self, name, env, policy_net, value_net, shared_layer, global_counter, discount_factor=0.99,
                 summary_writer=None, max_global_steps=None):
        super(SolowWorker, self).__init__(name, env, policy_net, value_net, shared_layer, global_counter,
                                          discount_factor, summary_writer, max_global_steps, 100.)

    def build_local_policy_net(self, policy_net, shared_layer):
        return GaussianPolicyEstimator(
            policy_net.num_actions, static_size=policy_net.static_size, temporal_size=policy_net.temporal_size,
            shared_layer=shared_layer,
        )

    def get_temporal_states(self, history):
        if len(history) == 1:
            return np.array(history[0]).reshape((1, -1))
        return np.array(history)

    def process_state(self, raw_state):
        return np.array([np.log(raw_state[0] / self.scale), raw_state[1]]).flatten()

    @staticmethod
    def get_random_action(mu, sigma, n_actions):
        raw_action = np.random.normal(mu, sigma)
        return [raw_action]

    @staticmethod
    def transform_raw_action(*raw_actions):
        return sigmoid(raw_actions[0])


class TradeWorker(GaussianWorker):

    def process_state(self, raw_state):
        new_states = []
        n_assets = len(raw_state) - 1

        new_states.append(np.log(raw_state[0] + 1e-4))

        quantity = raw_state[1:1 + n_assets]
        new_states += np.log(quantity + 1).tolist()
        prices = raw_state[1 + n_assets:]
        new_states += np.log(prices).tolist()

        return np.array(new_states).flatten()

    def get_temporal_states(self, history):
        return np.vstack(history)

    @staticmethod
    def get_random_action(mu, sigma, n_actions):
        raw_action = (mu + sigma * np.random.normal(size=(n_actions, ))).flatten()
        return TradeWorker.transform_raw_action(raw_action)

    @staticmethod
    def transform_raw_action(*raw_actions):
        return np.tanh(raw_actions[0])


class TickerGatedTraderWorker(GaussianWorker):

    def __init__(self, name, env, policy_net, value_net, shared_layer, global_counter, discount_factor=0.99,
                 summary_writer=None, max_global_steps=None, scale=1.):
        super(TickerGatedTraderWorker, self).__init__(name, env, policy_net, value_net, shared_layer,
                                                      global_counter, discount_factor, summary_writer,
                                                      max_global_steps, scale)
        self.n_assets = policy_net.num_assets

    def build_local_policy_net(self, policy_net, shared_layer):
        return DiscreteAndContPolicyEstimator(
            policy_net.num_assets, static_size=policy_net.static_size, temporal_size=policy_net.temporal_size,
            shared_layer=shared_layer
        )

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

    def get_temporal_states(self, history):
        return np.vstack(history)[:, 1 + self.n_assets:]

    @staticmethod
    def transform_raw_action(*actions):
        return [actions[0], sigmoid(actions[1])]

    @staticmethod
    def get_random_action(mu, sigma, choices):
        row_idx = np.arange(len(choices))
        mu = mu[row_idx, choices]
        sigma = sigma[row_idx, choices]
        return mu + sigma * np.random.normal(size=mu.shape)

    def get_action_from_policy(self, processed_state, history, session):
        preds = self._policy_net_predict(
            processed_state,
            self.get_temporal_states(history),
            session
        )
        probs = preds['probs'][0]
        self.debug.append(preds['mu'][0])
        discrete_action = self.get_random_discrete_action(probs)
        cont_action = self.get_random_action(preds['mu'][0], preds['sigma'][0], discrete_action)
        return [discrete_action, cont_action]

    def fill_feed_dict_for_update(self, states, temporal_state_matrix, advantages, actions, value_targets):
        discrete_actions, transformed_cont_actions = zip(*actions)
        return {
            self.policy_net.states: states,
            self.policy_net.history: temporal_state_matrix,
            self.policy_net.advantages: np.array(advantages[::-1]).flatten() / self.scale,
            self.policy_net.discrete_actions: discrete_actions,
            self.policy_net.actions: np.vstack(transformed_cont_actions),
            self.value_net.states: states,
            self.value_net.history: temporal_state_matrix,
            self.value_net.targets: value_targets.flatten(),
        }
