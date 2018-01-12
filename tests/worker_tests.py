import itertools

import numpy as np
import tensorflow as tf

from fed_gym.agents.a3c.worker import SolowWorker, TickerGatedTraderWorker
from fed_gym.agents.a3c.estimators import ValueEstimator, GaussianPolicyEstimator, rnn_graph_lstm, DiscreteAndContPolicyEstimator
from fed_gym.envs.fed_env import SolowEnv, TickerEnv


class SolowWorkerTest(tf.test.TestCase):

    def setUp(self):
        super(SolowWorkerTest, self).setUp()

        self.discount_factor = 0.99
        self.global_step = tf.Variable(0, name="global_step", trainable=False)
        self.global_counter = itertools.count()

        self.batch_size = 16
        self.num_actions = 1
        self.input_size = 2
        self.temporal_size = 2
        self.T = 10

        with tf.variable_scope("global"):
            self.global_policy_net = GaussianPolicyEstimator(
                self.num_actions, static_size=self.input_size, temporal_size=self.temporal_size,
                shared_layer=lambda x: rnn_graph_lstm(x, 32, 1, True)
            )
            self.global_value_net = ValueEstimator(
                static_size=self.input_size, temporal_size=self.temporal_size,
                shared_layer=lambda x: rnn_graph_lstm(x, 32, 1, True),
                reuse=True,
                num_actions=self.num_actions
            )

        self.shared_layer = lambda x: rnn_graph_lstm(x, 32, 1, True)

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
            state = SolowWorker.process_state(w.env.reset())
            temporal_state = w.get_temporal_states([state])
            mu, sig = w._policy_net_predict(state.flatten(), temporal_state.reshape((1, self.temporal_size)), sess)

            self.assertEqual(mu[0].shape, (self.num_actions, ))
            self.assertEqual(sig[0].shape, (self.num_actions, ))

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
            temporal_state = w.get_temporal_states([SolowWorker.process_state(state)])
            state_value = w._value_net_predict(state, temporal_state.reshape((1, self.temporal_size)), sess)
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
            w.history = [SolowWorker.process_state(w.state)]
            transitions, local_t, global_t, mus = w.run_n_steps(n_steps, sess)
            policy_net_loss, value_net_loss, policy_net_summaries, value_net_summaries, preds = w.update(transitions, sess)

        np.testing.assert_array_almost_equal(np.squeeze(preds['mu']), np.squeeze(mus[::-1]))
        self.assertEqual(len(transitions), n_steps)
        self.assertIsNotNone(policy_net_loss)
        self.assertIsNotNone(value_net_loss)
        self.assertIsNotNone(policy_net_summaries)
        self.assertIsNotNone(value_net_summaries)


class TickerTraderWorkerTests(tf.test.TestCase):

    def setUp(self):
        super(TickerTraderWorkerTests, self).setUp()

        self.discount_factor = 0.99
        self.global_step = tf.Variable(0, name="global_step", trainable=False)
        self.global_counter = itertools.count()

        self.batch_size = 16
        self.num_assets = 2
        self.num_actions = 3
        self.input_size = 1 + self.num_assets * 3 # cash + (quantity, price, vol) * n_assets
        self.temporal_size = self.num_assets * 2
        self.T = 10

        with tf.variable_scope("global"):
            self.global_policy_net = DiscreteAndContPolicyEstimator(
                self.num_assets, static_size=self.input_size, temporal_size=self.temporal_size,
                shared_layer=lambda x: rnn_graph_lstm(x, 32, 1, True)
            )
            self.global_value_net = ValueEstimator(
                static_size=self.input_size, temporal_size=self.temporal_size,
                shared_layer=lambda x: rnn_graph_lstm(x, 32, 1, True),
                reuse=True,
                num_actions=self.num_actions
            )

        self.shared_layer = lambda x: rnn_graph_lstm(x, 32, 1, True)

    def policy_predict_test(self):
        w = TickerGatedTraderWorker(
            name="test",
            env=TickerEnv(),
            policy_net=self.global_policy_net,
            value_net=self.global_value_net,
            shared_layer=self.shared_layer,
            global_counter=self.global_counter,
            discount_factor=self.discount_factor
        )

        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            state = TickerGatedTraderWorker.process_state(w.env.reset(), n_assets=self.num_assets)
            temporal_state = w.get_temporal_states([state], n_assets=self.num_assets)
            mu, sig, probs = w._policy_net_predict(state.flatten(), temporal_state.reshape((1, self.temporal_size)), sess)

            self.assertEqual(mu[0].shape, (self.num_assets, 3))
            self.assertEqual(sig[0].shape, (self.num_assets, 3))

    def value_predict_test(self):
        w = TickerGatedTraderWorker(
            name="test",
            env=TickerEnv(),
            policy_net=self.global_policy_net,
            value_net=self.global_value_net,
            shared_layer=self.shared_layer,
            global_counter=self.global_counter,
            discount_factor=self.discount_factor
        )

        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            state = w.env.reset()
            temporal_state = w.get_temporal_states(
                [TickerGatedTraderWorker.process_state(state, n_assets=self.num_assets)], n_assets=self.num_assets
            )
            state_value = w._value_net_predict(state, temporal_state.reshape((1, self.temporal_size)), sess)
            self.assertEqual(state_value.shape, ())

    def run_n_steps_and_update_test(self):

        n_steps = 10

        w = TickerGatedTraderWorker(
            name="test",
            env=TickerEnv(),
            policy_net=self.global_policy_net,
            value_net=self.global_value_net,
            shared_layer=self.shared_layer,
            global_counter=self.global_counter,
            discount_factor=self.discount_factor
        )

        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            w.state = w.env.reset()
            w.history = [TickerGatedTraderWorker.process_state(w.state, n_assets=self.num_assets)]
            transitions, local_t, global_t, mus = w.run_n_steps(n_steps, sess, max_seq_length=5)
            policy_net_loss, value_net_loss, policy_net_summaries, value_net_summaries, preds = w.update(
                transitions, sess, max_seq_length=5
            )

        np.testing.assert_array_almost_equal(np.squeeze(preds['mu']), np.squeeze(mus[::-1]))
        self.assertEqual(len(transitions), n_steps)
        self.assertIsNotNone(policy_net_loss)
        self.assertIsNotNone(value_net_loss)
        self.assertIsNotNone(policy_net_summaries)
        self.assertIsNotNone(value_net_summaries)
