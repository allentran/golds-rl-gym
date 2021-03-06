import itertools

import numpy as np
import tensorflow as tf

from fed_gym.agents.a3c.worker import SolowWorker, TickerGatedTraderWorker, GridSolowWorker
from fed_gym.agents.a3c.estimators import ValueEstimator, GaussianPolicyEstimator, rnn_graph_lstm, DiscreteAndContPolicyEstimator, DiscretePolicyEstimator
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
                shared_layer=lambda x_t, x: rnn_graph_lstm(x_t, x, 32, 1, True)
            )
            self.global_value_net = ValueEstimator(
                static_size=self.input_size, temporal_size=self.temporal_size,
                shared_layer=lambda x_t, x: rnn_graph_lstm(x_t, x, 32, 1, True),
                reuse=True,
                num_actions=self.num_actions
            )

        self.shared_layer = lambda x_t, x: rnn_graph_lstm(x_t, x, 32, 1, True)

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
            state = w.process_state(w.env.reset())
            temporal_state = w.state_processor.process_temporal_states([state])
            preds = w.policy_net.predict(state.flatten(), temporal_state.reshape((1, self.temporal_size)), sess)
            mu = preds['mu']
            sig = preds['sigma']

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
            temporal_state = w.state_processor.process_temporal_states([w.process_state(state)])
            state_value = w._value_net_predict(state, temporal_state.reshape((1, self.temporal_size)), sess)
            self.assertEqual(state_value.shape, ())

    def one_transition_test(self):

        n_steps = 1

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
            w.history = [w.process_state(w.state)]
            transitions, local_t, global_t, mus, done = w.run_n_steps(n_steps, sess, max_seq_length=5)
            transitions = [transitions[0]]
            w.update(transitions, sess, max_seq_length=5)

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
            w.history = [w.process_state(w.state)]

            transitions, local_t, global_t, mus, done = w.run_n_steps(n_steps, sess)
            policy_net_loss, value_net_loss, policy_net_summaries, value_net_summaries, preds = w.update(transitions, sess)
            np.testing.assert_array_almost_equal(np.squeeze(preds['mu']), np.squeeze(mus[::-1][:n_steps]))

            self.assertEqual(len(transitions), n_steps)
            self.assertIsNotNone(policy_net_loss)
            self.assertIsNotNone(value_net_loss)
            self.assertIsNotNone(policy_net_summaries)
            self.assertIsNotNone(value_net_summaries)

            transitions, local_t, global_t, mus, done = w.run_n_steps(n_steps, sess)
            policy_net_loss, value_net_loss, policy_net_summaries, value_net_summaries, preds = w.update(transitions, sess)
            np.testing.assert_array_almost_equal(np.squeeze(preds['mu']), np.squeeze(mus[::-1]))


class GridWorkerTests(tf.test.TestCase):

    def setUp(self):
        super(GridWorkerTests, self).setUp()

        self.discount_factor = 0.99
        self.global_step = tf.Variable(0, name="global_step", trainable=False)
        self.global_counter = itertools.count()

        self.batch_size = 16
        self.num_outputs = 1
        self.num_choices = 3
        self.input_size = 2
        self.temporal_size = 2
        self.T = 10

        with tf.variable_scope("global"):
            self.global_policy_net = DiscretePolicyEstimator(
                self.num_outputs, self.num_choices, static_size=self.input_size, temporal_size=self.temporal_size,
                shared_layer=lambda x_t, x: rnn_graph_lstm(x_t, x, 32, 1, True)
            )
            self.global_value_net = ValueEstimator(
                static_size=self.input_size, temporal_size=self.temporal_size,
                shared_layer=lambda x_t, x: rnn_graph_lstm(x_t, x, 32, 1, True),
                reuse=True,
            )

        self.shared_layer = lambda x_t, x: rnn_graph_lstm(x_t, x, 32, 1, True)

    def policy_predict_test(self):
        w = GridSolowWorker(
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
            state = w.state_processor.process_state(w.env.reset())
            temporal_state = w.state_processor.process_temporal_states([state])
            preds = w.policy_net.predict(state.flatten(), temporal_state.reshape((1, self.temporal_size)), sess)
            probs = preds['probs'][0]

            self.assertEqual(probs.shape, (self.num_outputs, self.num_choices))

    def value_predict_test(self):
        w = GridSolowWorker(
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
            temporal_state = w.state_processor.process_temporal_states([w.state_processor.process_state(state)])
            state_value = w._value_net_predict(state, temporal_state.reshape((1, self.temporal_size)), sess)
            self.assertEqual(state_value.shape, ())

    def run_n_steps_and_update_test(self):

        n_steps = 10

        w = GridSolowWorker(
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
            w.history = [w.state_processor.process_state(w.state)]

            transitions, local_t, global_t, probs, done = w.run_n_steps(n_steps, sess, max_seq_length=5)
            policy_net_loss, value_net_loss, policy_net_summaries, value_net_summaries, preds = w.update(
                transitions, sess, max_seq_length=5
            )
            np.testing.assert_array_almost_equal(np.squeeze(preds['probs']), np.squeeze(probs[::-1][:n_steps]))
            self.assertEqual(len(transitions), n_steps)
            self.assertIsNotNone(policy_net_loss)
            self.assertIsNotNone(value_net_loss)
            self.assertIsNotNone(policy_net_summaries)
            self.assertIsNotNone(value_net_summaries)

            transitions, local_t, global_t, probs, done = w.run_n_steps(n_steps, sess, max_seq_length=5)
            policy_net_loss, value_net_loss, policy_net_summaries, value_net_summaries, preds = w.update(
                transitions, sess, max_seq_length=5
            )
            np.testing.assert_array_almost_equal(np.squeeze(preds['probs']), np.squeeze(probs[::-1]))


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
                shared_layer=lambda x_t, x: rnn_graph_lstm(x_t, x, 32, 1, True)
            )
            self.global_value_net = ValueEstimator(
                static_size=self.input_size, temporal_size=self.temporal_size,
                shared_layer=lambda x_t, x: rnn_graph_lstm(x_t, x, 32, 1, True),
                reuse=True,
                num_actions=self.num_actions
            )

        self.shared_layer = lambda x_t, x: rnn_graph_lstm(x_t, x, 32, 1, True)

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
            state = w.state_processor.process_state(w.env.reset())
            temporal_state = w.state_processor.process_temporal_states([state])
            preds = w.policy_net.predict(state.flatten(), temporal_state.reshape((1, self.temporal_size)), sess)
            mu, sig, probs = preds['mu'], preds['sigma'], preds['probs']

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
            temporal_state = w.state_processor.process_temporal_states([w.state_processor.process_state(state)])
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
            w.history = [w.state_processor.process_state(w.state)]

            transitions, local_t, global_t, mus, done = w.run_n_steps(n_steps, sess, max_seq_length=5)
            policy_net_loss, value_net_loss, policy_net_summaries, value_net_summaries, preds = w.update(
                transitions, sess, max_seq_length=5
            )
            np.testing.assert_array_almost_equal(np.squeeze(preds['mu']), np.squeeze(mus[::-1][:n_steps]))
            self.assertEqual(len(transitions), n_steps)
            self.assertIsNotNone(policy_net_loss)
            self.assertIsNotNone(value_net_loss)
            self.assertIsNotNone(policy_net_summaries)
            self.assertIsNotNone(value_net_summaries)

            transitions, local_t, global_t, mus, done = w.run_n_steps(n_steps, sess, max_seq_length=5)
            policy_net_loss, value_net_loss, policy_net_summaries, value_net_summaries, preds = w.update(
                transitions, sess, max_seq_length=5
            )
            np.testing.assert_array_almost_equal(np.squeeze(preds['mu']), np.squeeze(mus[::-1]))

