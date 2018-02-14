import numpy as np
import tensorflow as tf

from fed_gym.agents.a3c.estimators import GaussianPolicyEstimator, ValueEstimator, rnn_graph_lstm, DiscreteAndContPolicyEstimator, DiscretePolicyEstimator
from fed_gym.agents.paac.policy_v_network import ConvPolicyVNetwork


class ConvNetworkTest(tf.test.TestCase):

    @classmethod
    def setUpClass(cls):
        super().setUpClass()

        cls.batch_size = 16
        cls.num_actions = 3
        cls.input_size = 5
        cls.temporal_size = 7
        cls.T = 11

        np.random.seed(1692)

        cls.states = np.random.random((cls.batch_size, cls.input_size))
        cls.temporal_states = np.random.random((cls.batch_size, cls.T, cls.temporal_size))
        cls.advantage = np.random.random((cls.batch_size, )).astype('float32')
        cls.discrete_actions = np.random.randint(0, cls.num_actions + 1, size=(cls.batch_size, 4)).astype('int32')

    def policy_predict_test(self):

        estimator = ConvPolicyVNetwork(
            {
                'name': 'test_conv_network',
                'num_actions': self.num_actions,
                'clip_norm': 40.,
                'clip_norm_type': 'global',
                'device': '/cpu:0',
                'static_size': None,
                'entropy_regularisation_strength': 0.,
                'scale': 1.,
                'height': 32,
                'width': 32,
                'channels': 3,
                'filters': 5,
                'conv_layers': 2
            }
        )

        action_idxs = np.hstack([
            np.arange(self.batch_size)[:, None],
            np.random.randint(0, estimator.height, self.batch_size)[:, None],
            np.random.randint(0, estimator.width, self.batch_size)[:, None],
            np.random.randint(0, self.num_actions, self.batch_size)[:, None],
        ])
        state = np.random.uniform(
            0., 1.,
            (self.batch_size, estimator.height, estimator.width, estimator.channels),
        )
        history = np.random.uniform(
            0., 1.,
            (self.batch_size, self.T, estimator.height, estimator.width, estimator.channels),
        )
        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())

            feed_dict = {
                estimator.state: state,
                estimator.history: history,
                estimator.advantages: np.ones_like(self.advantage),
                estimator.action_idxs: action_idxs,
                estimator.critic_target: np.ones_like(self.advantage)
            }
            pred = sess.run(
                {
                    'probs': estimator.probs,
                    'policy_loss': estimator.policy_loss,
                    'vs': estimator.vs,
                    'critic_loss': estimator.critic_loss
                },
                feed_dict
            )

        self.assertEqual(pred['probs'].shape, (self.batch_size, self.num_actions))
        np.testing.assert_almost_equal(pred['probs'], 0.33, decimal=1)
        self.assertEqual(pred['policy_loss'].shape, ())
        self.assertEqual(pred['vs'].shape, (self.batch_size, ))
        self.assertEqual(pred['critic_loss'].shape, (self.batch_size, ))

class DiscretePolicyEstimatorTest(tf.test.TestCase):

    @classmethod
    def setUpClass(cls):
        super(DiscretePolicyEstimatorTest, cls).setUpClass()

        cls.batch_size = 16
        cls.num_actions = 3
        cls.input_size = 5
        cls.temporal_size = 7
        cls.n_assets = 2
        cls.T = 10

        np.random.seed(1692)

        cls.states = np.random.random((cls.batch_size, cls.input_size))
        cls.temporal_states = np.random.random((cls.batch_size, cls.T, cls.temporal_size))
        cls.advantage = np.random.random((cls.batch_size, )).astype('float32')
        cls.discrete_actions = np.random.randint(0, 3, size=(cls.batch_size, cls.n_assets)).astype('int32')

    def learn_policy_test(self):
        tf.Variable(0, name='global_step',trainable=False)
        estimator = DiscretePolicyEstimator(
            self.n_assets, self.num_actions, static_size=self.input_size, temporal_size=self.temporal_size,
            shared_layer=lambda x_t, x: rnn_graph_lstm(x_t, x, 32, 1, True),
            seed=1692,
            learning_rate=1e-3
        )

        def all_idx(idx, axis):
            grid = np.ogrid[tuple(map(slice, idx.shape))]
            grid.insert(axis, idx)
            return tuple(grid)

        grads = [g for g, _ in estimator.grads_and_vars]

        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())

            # Run feeds
            for _ in range(1000):
                feed_dict = {
                    estimator.states: self.states,
                    estimator.history: self.temporal_states,
                    estimator.advantages: np.ones_like(self.advantage),
                    estimator.actions: self.discrete_actions
                }
                pred = sess.run(estimator.predictions, feed_dict)

                grads_ = sess.run(grads, feed_dict)

                grad_feed_dict = { k: v for k, v in zip(grads, grads_) }
                _ = sess.run(estimator.train_op, grad_feed_dict)

        # index 3D probs with 2D array of choices
        prob_optimal_choice = pred['probs'][all_idx(self.discrete_actions, 2)]
        self.assertLess(0.9, prob_optimal_choice.mean())


class GatedPolicyEstimatorTest(tf.test.TestCase):

    @classmethod
    def setUpClass(cls):
        super(GatedPolicyEstimatorTest, cls).setUpClass()

        cls.batch_size = 16
        cls.num_actions = 3
        cls.input_size = 5
        cls.temporal_size = 7
        cls.n_assets = 2
        cls.T = 10

        np.random.seed(1692)

        cls.states = np.random.random((cls.batch_size, cls.input_size))
        cls.temporal_states = np.random.random((cls.batch_size, cls.T, cls.temporal_size))
        cls.advantage = np.random.random((cls.batch_size, )).astype('float32')
        cls.actions = np.random.random((cls.batch_size, cls.n_assets)).astype('float32')
        cls.discrete_actions = np.random.randint(0, 3, size=(cls.batch_size, cls.n_assets)).astype('int32')

    def learn_policy_test(self):
        tf.Variable(0, name='global_step',trainable=False)
        estimator = DiscreteAndContPolicyEstimator(
            self.n_assets, static_size=self.input_size, temporal_size=self.temporal_size,
            shared_layer=lambda x_t, x: rnn_graph_lstm(x_t, x, 32, 1, True),
            seed=1692,
            learning_rate=1e-3
        )

        def all_idx(idx, axis):
            grid = np.ogrid[tuple(map(slice, idx.shape))]
            grid.insert(axis, idx)
            return tuple(grid)

        grads = [g for g, _ in estimator.grads_and_vars]

        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())

            # Run feeds
            for _ in range(1000):
                feed_dict = {
                    estimator.states: self.states,
                    estimator.history: self.temporal_states,
                    estimator.advantages: np.ones_like(self.advantage),
                    estimator.actions: self.actions,
                    estimator.discrete_actions: self.discrete_actions
                }
                pred = sess.run(estimator.predictions, feed_dict)

                grads_ = sess.run(grads, feed_dict)

                grad_feed_dict = { k: v for k, v in zip(grads, grads_) }
                _ = sess.run(estimator.train_op, grad_feed_dict)
                cont_action_optimal_choice = pred['mu'][all_idx(self.discrete_actions, 2)]

        # index 3D probs with 2D array of choices
        prob_optimal_choice = pred['probs'][all_idx(self.discrete_actions, 2)]
        cont_action_optimal_choice = pred['mu'][all_idx(self.discrete_actions, 2)]

        self.assertLess(0.9, prob_optimal_choice.mean())
        self.assertLess(np.mean(np.abs((cont_action_optimal_choice - self.actions))), 0.2)

    def predict_test(self):
        global_step = tf.Variable(0, name='global_step',trainable=False)
        estimator = DiscreteAndContPolicyEstimator(
            self.n_assets, static_size=self.input_size, temporal_size=self.temporal_size,
            shared_layer=lambda x_t, x: rnn_graph_lstm(x_t, x, 32, 1, True)
        )

        grads = [g for g, _ in estimator.grads_and_vars]

        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())

            # Run feeds
            losses = []
            for _ in range(100):
                feed_dict = {
                    estimator.states: self.states,
                    estimator.history: self.temporal_states,
                    estimator.advantages: self.advantage,
                    estimator.actions: self.actions,
                    estimator.discrete_actions: self.discrete_actions
                }
                loss = sess.run(estimator.loss, feed_dict)
                losses.append(loss)
                pred = sess.run(estimator.predictions, feed_dict)

                grads_ = sess.run(grads, feed_dict)

                grad_feed_dict = { k: v for k, v in zip(grads, grads_) }
                _ = sess.run(estimator.train_op, grad_feed_dict)

            # Assertions
            np.testing.assert_array_less(0., pred['sigma'])
            self.assertEqual(pred['probs'].shape[0], self.batch_size)
            self.assertEqual(pred['probs'].shape[1], self.n_assets)
            self.assertEqual(pred['probs'].shape[2], self.num_actions)
            self.assertEqual(pred['mu'].shape[1], self.n_assets)
            self.assertEqual(pred['mu'].shape[2], self.num_actions)
            self.assertEqual(pred['sigma'].shape[1], self.n_assets)
            self.assertEqual(pred['sigma'].shape[2], self.num_actions)


class PolicyEstimatorTest(tf.test.TestCase):

    @classmethod
    def setUpClass(cls):
        super(PolicyEstimatorTest, cls).setUpClass()

        cls.batch_size = 16
        cls.num_actions = 3
        cls.input_size = 5
        cls.temporal_size = 7
        cls.T = 10

        np.random.seed(1692)

        cls.states = np.random.random((cls.batch_size, cls.input_size))
        cls.temporal_states = np.random.random((cls.batch_size, cls.T, cls.temporal_size))
        cls.advantage = np.random.random((cls.batch_size, )).astype('float32')
        cls.actions = np.random.random((cls.batch_size, cls.num_actions)).astype('float32')

    def learn_policy_test(self):

        global_step = tf.Variable(0, name='global_step',trainable=False)
        estimator = GaussianPolicyEstimator(
            self.num_actions, static_size=self.input_size, temporal_size=self.temporal_size,
            shared_layer=lambda x_t, x: rnn_graph_lstm(x_t, x, 32, 1, True),
            learning_rate=1e-3,
            seed=1692
        )

        grads = [g for g, _ in estimator.grads_and_vars]

        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())

            # Run feeds
            for _ in range(1000):
                feed_dict = {
                    estimator.states: self.states,
                    estimator.history: self.temporal_states,
                    estimator.advantages: np.ones_like(self.advantage),
                    estimator.actions: self.actions
                }
                pred = sess.run(estimator.predictions, feed_dict)

                grads_ = sess.run(grads, feed_dict)

                grad_feed_dict = { k: v for k, v in zip(grads, grads_) }
                _ = sess.run(estimator.train_op, grad_feed_dict)

        self.assertLess(np.mean(np.abs((pred['mu'] - self.actions))), 0.1)

    def gaussian_predict_test(self):
        global_step = tf.Variable(0, name='global_step',trainable=False)
        estimator = GaussianPolicyEstimator(
            self.num_actions, static_size=self.input_size, temporal_size=self.temporal_size,
            shared_layer=lambda x_t, x: rnn_graph_lstm(x_t, x, 32, 1, True)
        )

        grads = [g for g, _ in estimator.grads_and_vars]

        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())

            # Run feeds
            losses = []
            for _ in range(10):
                feed_dict = {
                    estimator.states: self.states,
                    estimator.history: self.temporal_states,
                    estimator.advantages: self.advantage,
                    estimator.actions: self.actions
                }
                loss = sess.run(estimator.loss, feed_dict)
                losses.append(loss)
                pred = sess.run(estimator.predictions, feed_dict)

                grads_ = sess.run(grads, feed_dict)

                grad_feed_dict = { k: v for k, v in zip(grads, grads_) }
                _ = sess.run(estimator.train_op, grad_feed_dict)

            # Assertions
            self.assertLess(losses[-1], losses[0])
            np.testing.assert_array_less(0., pred['sigma'])
            self.assertEqual(pred['mu'].shape[1], self.num_actions)
            self.assertEqual(pred['sigma'].shape[1], self.num_actions)


class ValueEstimatorTest(tf.test.TestCase):

    @classmethod
    def setUpClass(cls):
        super(ValueEstimatorTest, cls).setUpClass()

        cls.batch_size = 16
        cls.num_actions = 3
        cls.input_size = 5
        cls.temporal_size = 7
        cls.T = 10

        cls.states = np.random.random((cls.batch_size, cls.input_size))
        cls.temporal_states = np.random.random((cls.batch_size, cls.T, cls.temporal_size))
        cls.targets = np.random.random((cls.batch_size, )).astype('float32')

    def predict_test(self):
        global_step = tf.Variable(0, name='global_step',trainable=False)
        estimator = ValueEstimator(
            static_size=self.input_size, temporal_size=self.temporal_size,
            shared_layer=lambda x_t, x: rnn_graph_lstm(x_t, x, 32, 1, True),
            learning_rate=1e-3
        )

        grads = [g for g, _ in estimator.grads_and_vars]

        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())

            # Run feeds
            feed_dict = {
                estimator.states: self.states,
                estimator.history: self.temporal_states,
                estimator.targets: self.targets
            }
            losses = []
            for _ in range(1000):
                loss = sess.run(estimator.loss, feed_dict)
                pred = sess.run(estimator.predictions, feed_dict)
                grads_ = sess.run(grads, feed_dict)

                grad_feed_dict = { k: v for k, v in zip(grads, grads_) }
                _ = sess.run(estimator.train_op, grad_feed_dict)
                losses.append(loss)

            # Assertions
            self.assertLess(loss, 1e-1)
            self.assertGreater(loss, 0.)
            self.assertEqual(pred['logits'].shape, (self.batch_size, ))
            self.assertLess(losses[-1], losses[0])
