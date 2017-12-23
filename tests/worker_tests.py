import itertools
import numpy as np
import tensorflow as tf

from fed_gym.agents.a3c.worker import SolowWorker
from fed_gym.agents.a3c.estimators import ValueEstimator, GaussianPolicyEstimator, rnn_graph_lstm
from fed_gym.envs.fed_env import SolowEnv


class WorkerTest(tf.test.TestCase):

    def setUp(self):
        super(WorkerTest, self).setUp()

        self.discount_factor = 0.99
        self.global_step = tf.Variable(0, name="global_step", trainable=False)
        self.global_counter = itertools.count()

        self.batch_size = 16
        self.num_actions = 1
        self.input_size = 2
        self.temporal_size = 1
        self.T = 10

        with tf.variable_scope("global") as vs:
            self.global_policy_net = GaussianPolicyEstimator(
                self.num_actions, static_size=self.input_size, temporal_size=self.temporal_size,
                shared_layer=lambda x: rnn_graph_lstm(x, 32, 2, True)
            )
            self.global_value_net = ValueEstimator(
                static_size=self.input_size, temporal_size=self.temporal_size,
                shared_layer=lambda x: rnn_graph_lstm(x, 32, 2, True),
                reuse=True
            )

        self.shared_layer = lambda x: rnn_graph_lstm(x, 32, 2, True)

    def policy_predict_test(self):
        w = SolowWorker(
            name="test",
            env=SolowEnv(),
            policy_net=self.global_policy_net,
            value_net=self.global_value_net,
            shared_layer=self.shared_layer,
            global_counter=self.global_counter,
            discount_factor=self.discount_factor
        )

        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            state = w.env.reset()
            temporal_state = w.get_temporal_states([state])
            mu, sig = w._policy_net_predict(state.flatten(), temporal_state.reshape((1, self.temporal_size)), sess)

            self.assertEqual(mu.shape, (self.num_actions, ))
            self.assertEqual(sig.shape, (self.num_actions, ))

    def value_predict_test(self):
        w = SolowWorker(
            name="test",
            env=SolowEnv(),
            policy_net=self.global_policy_net,
            value_net=self.global_value_net,
            shared_layer=self.shared_layer,
            global_counter=self.global_counter,
            discount_factor=self.discount_factor
        )

        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            state = w.env.reset()
            temporal_state = w.get_temporal_states([state])
            state_value = w._value_net_predict(state, temporal_state.reshape((1, 1)), sess)
            self.assertEqual(state_value.shape, ())

    def run_n_steps_and_update_test(self):

        n_steps = 10

        w = SolowWorker(
            name="test",
            env=SolowEnv(),
            policy_net=self.global_policy_net,
            value_net=self.global_value_net,
            shared_layer=self.shared_layer,
            global_counter=self.global_counter,
            discount_factor=self.discount_factor
        )

        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            w.state = w.env.reset()
            w.history = [w.state]
            transitions, local_t, global_t = w.run_n_steps(n_steps, sess)
            policy_net_loss, value_net_loss, policy_net_summaries, value_net_summaries = w.update(transitions, sess)

        self.assertEqual(len(transitions), n_steps)
        self.assertIsNotNone(policy_net_loss)
        self.assertIsNotNone(value_net_loss)
        self.assertIsNotNone(policy_net_summaries)
        self.assertIsNotNone(value_net_summaries)
