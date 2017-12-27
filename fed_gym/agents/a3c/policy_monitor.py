import os
import numpy as np
import tensorflow as tf
import time

from gym.wrappers import Monitor

from estimators import GaussianPolicyEstimator, rnn_graph_lstm
from worker import make_copy_params_op, SolowWorker


class PolicyMonitor(object):
    """
    Helps evaluating a policy by running an episode in an environment,
    saving a video, and plotting summaries to Tensorboard.

    Args:
      env: environment to run in
      policy_net: A policy estimator
      summary_writer: a tf.train.SummaryWriter used to write Tensorboard summaries
    """
    def __init__(self, env, policy_net, summary_writer, saver=None, num_actions=None, input_size=None, temporal_size=None):

        self.video_dir = os.path.join(summary_writer.get_logdir(), "../videos")
        self.video_dir = os.path.abspath(self.video_dir)

        self.env = Monitor(env, directory=os.path.abspath(summary_writer.get_logdir()), resume=True)
        self.global_policy_net = policy_net
        self.summary_writer = summary_writer
        self.saver = saver

        self.checkpoint_path = os.path.abspath(os.path.join(summary_writer.get_logdir(), "../checkpoints/model"))
        if not os.path.exists(self.video_dir):
            os.makedirs(self.video_dir)

        # Local policy net
        with tf.variable_scope("policy_eval"):
            self.policy_net = GaussianPolicyEstimator(
                num_actions, static_size=input_size, temporal_size=temporal_size,
                shared_layer=lambda x: rnn_graph_lstm(x, 32, 2, True)
            )

        # Op to copy params from global policy/value net parameters
        self.copy_params_op = make_copy_params_op(
            tf.contrib.slim.get_variables(scope="global", collection=tf.GraphKeys.TRAINABLE_VARIABLES),
            tf.contrib.slim.get_variables(scope="policy_eval", collection=tf.GraphKeys.TRAINABLE_VARIABLES))

    def _policy_net_predict(self, state, history, sess):
        feed_dict = {
            self.policy_net.states: [state],
            self.policy_net.history: [history],
        }
        preds = sess.run(self.policy_net.predictions, feed_dict)
        return preds["mu"][0], preds["sigma"][0]

    def eval_once(self, sess, max_steps=None):
        with sess.as_default(), sess.graph.as_default():
            # Copy params to local model
            global_step, _ = sess.run([tf.train.get_global_step(), self.copy_params_op])

            # Run an episode
            done = False
            state = self.env.reset()
            history = SolowWorker.get_temporal_states([state])
            total_reward = 0.0
            episode_length = 0
            while not done:
                if max_steps and episode_length > max_steps:
                    break
                mu, sig = self._policy_net_predict(state, history, sess)
                action = SolowWorker.get_random_action(mu, sig)
                next_state, reward, done, _ = self.env.step(action)
                total_reward += reward
                episode_length += 1
                state = next_state

            # Add summaries
            episode_summary = tf.Summary()
            episode_summary.value.add(simple_value=total_reward, tag="eval/total_reward")
            episode_summary.value.add(simple_value=episode_length, tag="eval/episode_length")
            self.summary_writer.add_summary(episode_summary, global_step)
            self.summary_writer.flush()

            if self.saver is not None:
                self.saver.save(sess, self.checkpoint_path)

            tf.logging.info(
                "Eval results at step {}: total_reward {}, episode_length {}".format(
                    global_step, total_reward, episode_length
                )
            )

            return total_reward, episode_length

    def continuous_eval(self, eval_every, sess, coord, max_steps=None):
        """
        Continuously evaluates the policy every [eval_every] seconds.
        """
        try:
            while not coord.should_stop():
                self.eval_once(sess, max_steps=max_steps)
                # Sleep until next evaluation cycle
                time.sleep(eval_every)
        except tf.errors.CancelledError:
            return
