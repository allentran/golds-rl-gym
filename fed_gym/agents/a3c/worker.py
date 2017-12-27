import itertools
import collections
import numpy as np
import tensorflow as tf

from estimators import ValueEstimator, GaussianPolicyEstimator

Transition = collections.namedtuple(
    "Transition",
    ["state", "action", "reward", "next_state", "done"]
)


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
                reuse=True
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
        self.seq_lengths = None

    def run(self, sess, coord, t_max):
        with sess.as_default(), sess.graph.as_default():
            # Initial state
            self.state = self.env.reset()
            self.history.append(self.state)

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
                    self.update(transitions, sess)

            except tf.errors.CancelledError:
                return

    def _policy_net_predict(self, state, history, sess):
        feed_dict = {
            self.policy_net.states: [state],
            self.policy_net.history: [history],
        }
        preds = sess.run(self.policy_net.predictions, feed_dict)
        return preds["mu"][0], preds["sigma"][0]

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
    def get_random_action(mu, sigma):
        raise NotImplementedError

    def run_n_steps(self, n, sess):
        transitions = []
        for _ in xrange(n):
            # Take a step
            action_mu, action_sigma = self._policy_net_predict(
                self.state, self.get_temporal_states(self.history), sess
            )
            action = self.get_random_action(action_mu, action_sigma)
            next_state, reward, done, _ = self.env.step(action)

            # Store transition
            transitions.append(Transition(
                state=self.state,
                action=action,
                reward=reward,
                next_state=next_state,
                done=done)
            )

            # Increase local and global counters
            local_t = next(self.local_counter)
            global_t = next(self.global_counter)

            if local_t % 100 == 0:
                tf.logging.info("{}: local Step {}, global step {}".format(self.name, local_t, global_t))

            if done:
                self.state = self.env.reset()
                self.history.append(self.state)
                break
            else:
                self.state = next_state
                self.history.append(next_state)

        return transitions, local_t, global_t

    def update(self, transitions, sess):
        """
        Updates global policy and value networks based on collected experience

        Args:
          transitions: A list of experience transitions
          sess: A Tensorflow session
        """

        # If we episode was not done we bootstrap the value from the last state
        reward = 0.0
        history = self.history
        if not transitions[-1].done:
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
            state = transition.state
            history_t = history[:-(idx + 1)]
            policy_advantage = reward - self._value_net_predict(state, self.get_temporal_states(history_t), sess)
            # Accumulate updates
            temporal_states.append(self.get_temporal_states(history_t))
            states.append(transition.state)
            actions.append(transition.action)
            policy_advantages.append(policy_advantage)
            value_targets.append(reward)

        temporal_state_matrix = tf.keras.preprocessing.sequence.pad_sequences(
            temporal_states, dtype='float32', padding='post'
        )

        feed_dict = {
            self.policy_net.states: np.array(states),
            self.policy_net.history: temporal_state_matrix,
            self.policy_net.advantages: policy_advantages,
            self.policy_net.actions: np.array(actions).reshape((-1, self.global_policy_net.num_actions)),
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


class SolowWorker(Worker):

    @staticmethod
    def get_temporal_states(history):
        return np.array(history)[:, 1][:, None]

    @staticmethod
    def get_random_action(mu, sigma):
        raw_action = np.random.normal(mu, sigma, 1)[0]
        return 1 / (1 + np.exp(-raw_action))
