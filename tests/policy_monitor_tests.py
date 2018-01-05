import gym
import sys
import os
import unittest
import tensorflow as tf
import tempfile

from fed_gym.agents.a3c.policy_monitor import PolicyMonitor
from fed_gym.agents.a3c.estimators import ValueEstimator, GaussianPolicyEstimator, rnn_graph_lstm


def make_env():
    return gym.envs.make("Solow-v0")


class PolicyMonitorTest(tf.test.TestCase):
    def setUp(self):
        super(PolicyMonitorTest, self).setUp()

        self.batch_size = 16
        self.num_actions = 1
        self.input_size = 2
        self.temporal_size = 1
        self.T = 10

        self.env = make_env()
        self.global_step = tf.Variable(0, name="global_step", trainable=False)
        self.summary_writer = tf.summary.FileWriter(tempfile.mkdtemp())

        with tf.variable_scope("global"):
            self.global_policy_net = GaussianPolicyEstimator(
                self.num_actions, static_size=self.input_size, temporal_size=self.temporal_size,
                shared_layer=lambda x: rnn_graph_lstm(x, 32, 2, True)
            )
            self.global_value_net = ValueEstimator(
                static_size=self.input_size, temporal_size=self.temporal_size,
                shared_layer=lambda x: rnn_graph_lstm(x, 32, 2, True),
                reuse=True
            )

    def testEvalOnce(self):
        pe = PolicyMonitor(
            env=self.env,
            policy_net=self.global_policy_net,
            summary_writer=self.summary_writer,
            num_actions=self.num_actions,
            input_size=self.input_size,
            temporal_size=self.temporal_size
        )

        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            total_reward, episode_length = pe.eval_once(sess)
            self.assertTrue(episode_length > 10)
