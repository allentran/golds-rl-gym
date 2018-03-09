import os
import json
import queue
import time

import numpy as np
import tensorflow as tf
from gym.wrappers import Monitor

from .policy_v_network import ConvSingleAgentPolicyNetwork, FlatPolicyVNetwork
from.emulator_runner import SwarmRunner, SolowRunner
from ..a3c.worker import make_copy_params_op


class PolicyMonitor(object):
    """
    Helps evaluating a policy by running an episode in an environment,
    saving a video, and plotting summaries to Tensorboard.

    Args:
      env: environment to run in
      policy_net: A policy estimator
      summary_writer: a tf.train.SummaryWriter used to write Tensorboard summaries
    """
    def __init__(self, env, global_policy_net, state_processor, summary_writer, saver=None, network_conf=None):

        self.env = Monitor(env, directory=os.path.abspath(summary_writer.get_logdir()), resume=True)
        self.state_processor = state_processor
        self.global_policy_net = global_policy_net
        self.summary_writer = summary_writer
        self.saver = saver
        self.best_score = - np.inf

        self.checkpoint_path = os.path.abspath(os.path.join(summary_writer.get_logdir(), "../checkpoints/model"))

        # Local policy net
        with tf.variable_scope("policy_eval"):
            self.policy_net = self._create_policy_estimator(network_conf)

        # Op to copy params from global policy/value net parameters
        self.copy_params_op = make_copy_params_op(
            tf.contrib.slim.get_variables(scope="global", collection=tf.GraphKeys.TRAINABLE_VARIABLES),
            tf.contrib.slim.get_variables(scope="policy_eval", collection=tf.GraphKeys.TRAINABLE_VARIABLES))

    def get_action_from_policy(self, processed_state, history, positions, sess):
        predictions = self.policy_net.predict(processed_state, history, positions, sess)
        mu, sigma = predictions['mu'], predictions['sigma']
        new_actions = mu + sigma * np.random.normal(size=mu.shape)
        return new_actions

    @staticmethod
    def _create_policy_estimator(conf):
        raise NotImplementedError

    def eval_once(self, sess, max_sequence_length=5):
        raise NotImplementedError

    def continuous_eval(self, eval_every, sess, coord, max_seq_length):
        """
        Continuously evaluates the policy every [eval_every] seconds.
        """
        total_rewards = []
        episode_lengths = []
        try:
            while not coord.should_stop():
                total_reward, episode_length, rewards = self.eval_once(sess, max_sequence_length=max_seq_length)
                total_rewards.append(total_reward)
                episode_lengths.append(episode_length)
                time.sleep(eval_every)
        except tf.errors.CancelledError:
            return


class SolowPolicyMonitor(PolicyMonitor):

    def get_action_from_policy(self, processed_state, history, positions, sess):
        raw_actions = super().get_action_from_policy(processed_state, history, positions, sess)
        return SolowRunner.transform_actions_for_env(raw_actions)

    @staticmethod
    def _create_policy_estimator(conf):
        return FlatPolicyVNetwork(conf)

    def eval_once(self, sess, max_sequence_length=5):
        with sess.as_default(), sess.graph.as_default():
            global_step, _ = sess.run([self.global_policy_net.global_step_tensor, self.copy_params_op])
            histories = []

            # Run an episode
            done = False
            state = self.env.reset()
            processed_state = self.state_processor.process_state(state)
            histories.append(np.array(processed_state))
            history = np.array([histories])
            total_reward = 0.0
            episode_length = 0
            rewards = []
            while not done:
                action = self.get_action_from_policy(np.array([processed_state]), history, None, sess)
                next_state, reward, done, _ = self.env.step(action)
                processed_state = self.state_processor.process_state(next_state)
                histories.append(np.array(processed_state))
                history = np.array([histories[-max_sequence_length:]])
                total_reward += reward
                episode_length += 1
                rewards.append(reward)

                histories = histories[-2 * max_sequence_length:]

            # Add summaries
            episode_summary = tf.Summary()
            episode_summary.value.add(simple_value=total_reward, tag="eval/total_reward")
            episode_summary.value.add(simple_value=episode_length, tag="eval/episode_length")
            self.summary_writer.add_summary(episode_summary, global_step)
            self.summary_writer.flush()

            # if self.saver is not None:
            #     self.saver.save(sess, self.checkpoint_path)

            tf.logging.info(
                "Eval results at step {}: avg_reward {}, std_reward {}, episode_length {}".format(
                    global_step, np.mean(rewards), np.std(rewards), episode_length
                )
            )

            return total_reward, episode_length, rewards


class SwarmPolicyMonitor(PolicyMonitor):

    def get_action_from_policy(self, processed_state, history, positions, sess):
        predictions = self.policy_net.predict(processed_state, sess)
        mu, sigma = predictions['mu'], predictions['sigma']
        raw_actions = mu + sigma * np.random.normal(size=mu.shape)
        return SwarmRunner.transform_actions_for_env(raw_actions)

    @staticmethod
    def _create_policy_estimator(conf):
        return ConvSingleAgentPolicyNetwork(conf)

    @staticmethod
    def _save_actions(score, actions):
        with open('swarm-eval.json', 'w') as f:
            json.dump(
                {
                    'score': score,
                    'actions': actions
                },
                f
            )

    def eval_once(self, sess, max_sequence_length=5, actions: queue.Queue=None):
        with sess.as_default(), sess.graph.as_default():
            # Copy params to local model
            global_step, _ = sess.run([self.global_policy_net.global_step_tensor, self.copy_params_op])
            histories = []

            # Run an episode
            done = False
            state = self.env.reset()
            processed_state = self.state_processor.process_state(state)
            processed_state = SwarmRunner.get_local_states(processed_state, self.state_processor.positions)
            histories.append(np.array(processed_state))
            history = np.array(histories)
            history = np.swapaxes(history, 0, 1)
            total_reward = 0.0
            episode_length = 0
            rewards = []
            taken_actions = []
            while not done:
                if not actions:
                    action = self.get_action_from_policy(np.array(processed_state), history, self.state_processor.positions, sess)
                else:
                    action = actions.get()
                taken_actions.append(action.tolist())
                next_state, reward, done, _ = self.env.step(action)
                processed_state = self.state_processor.process_state(next_state)
                processed_state = np.array(SwarmRunner.get_local_states(processed_state, self.state_processor.positions))
                histories.append(np.array(processed_state))
                history = np.array(histories[-max_sequence_length:])
                history = np.swapaxes(history, 0, 1)
                total_reward += reward
                episode_length += 1
                rewards.append(reward)

                histories = histories[-2 * max_sequence_length:]

            if total_reward > self.best_score:
                self.best_score = total_reward
                self._save_actions(total_reward, taken_actions)

            # Add summaries
            episode_summary = tf.Summary()
            episode_summary.value.add(simple_value=total_reward, tag="eval/total_reward")
            episode_summary.value.add(simple_value=episode_length, tag="eval/episode_length")
            self.summary_writer.add_summary(episode_summary, global_step)
            self.summary_writer.flush()

            # if self.saver is not None:
            #     self.saver.save(sess, self.checkpoint_path)

            tf.logging.info(
                "Eval results at step {}: avg_reward {}, std_reward {}, episode_length {}".format(
                    global_step, np.mean(rewards), np.std(rewards), episode_length
                )
            )

            return total_reward, episode_length, rewards
