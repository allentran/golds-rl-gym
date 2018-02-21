import gym
import numpy as np
import itertools
import tensorflow as tf
import tempfile

from fed_gym.agents.a3c.policy_monitor import PolicyMonitor
from fed_gym.agents.a3c.estimators import ValueEstimator, GaussianPolicyEstimator, rnn_graph_lstm
from fed_gym.agents.state_processors import SolowStateProcessor
from fed_gym.agents.a3c.worker import SolowWorker


def make_env():
    return gym.envs.make("SolowSS-v0")


class PolicyMonitorTest(tf.test.TestCase):
    def setUp(self):
        super(PolicyMonitorTest, self).setUp()

        self.batch_size = 16
        self.num_actions = 1
        self.input_size = 2
        self.temporal_size = 2
        self.T = 10

        self.env = make_env()
        self.global_step = tf.Variable(0, name="global_step", trainable=False)
        self.summary_writer = tf.summary.FileWriter(tempfile.mkdtemp())

        with tf.variable_scope("global"):
            self.global_policy_net = GaussianPolicyEstimator(
                self.num_actions, static_size=self.input_size, temporal_size=self.temporal_size,
                shared_layer=lambda x_t, x: rnn_graph_lstm(x_t, x, 32, 1, True)
            )
            self.global_value_net = ValueEstimator(
                static_size=self.input_size, temporal_size=self.temporal_size,
                shared_layer=lambda x_t, x: rnn_graph_lstm(x_t, x, 32, 1, True),
                reuse=True
            )

    def testEvalOnce(self):
        pe = PolicyMonitor(
            env=make_env(),
            state_processor=SolowStateProcessor(),
            global_policy_net=self.global_policy_net,
            summary_writer=self.summary_writer,
            num_actions=self.num_actions,
            input_size=self.input_size,
            temporal_size=self.temporal_size
        )

        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            total_reward, episode_length, rewards = pe.eval_once(sess)
            self.assertTrue(episode_length > 10)

    def policy_monitor_worker_equal(self):

        global_counter = itertools.count()
        worker_env = make_env()
        worker_env.seed(1692)
        worker = SolowWorker(
            'test_worker',
            env=worker_env,
            policy_net=self.global_policy_net,
            value_net=None,
            shared_layer=lambda x_t, x: rnn_graph_lstm(x_t, x, 32, 1, True),
            global_counter=global_counter,
        )

        env = make_env()
        pe = PolicyMonitor(
            env=env,
            state_processor=SolowStateProcessor(),
            global_policy_net=self.global_policy_net,
            summary_writer=self.summary_writer,
            num_actions=self.num_actions,
            input_size=self.input_size,
            temporal_size=self.temporal_size
        )

        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())

            worker.state = worker_env.reset()
            worker.history.append(worker.process_state(worker.state))

            sess.run(worker.copy_params_op)

            transitions = worker.run_n_steps(10, sess, stochastic=False)
            worker_rewards = [t.reward for t in transitions[0]]

            pe.env = make_env()
            pe.env.seed(1692)
            pe.policy_net = worker.policy_net
            total_reward, episode_length, rewards = pe.eval_once(sess)
            monitor_rewards = rewards[:10]

        np.testing.assert_almost_equal(monitor_rewards, worker_rewards, decimal=4)
