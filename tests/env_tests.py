import unittest

import numpy as np

from fed_gym.envs import fed_env


class TradingEnvTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        super(TradingEnvTests, cls).setUpClass()
        cls.env = fed_env.TradeEnv()

    def deplete_test(self):
        self.env.reset()

        for _ in xrange(100):
            (cash, quantity, price_history), reward, done, _ = self.env.step(np.array([0.1, 0.1]))

        self.assertTrue(done)
        self.assertLessEqual(cash, self.env.MIN_CASH)
        np.testing.assert_array_less(0, quantity)
        np.testing.assert_array_less(0, price_history)

    def buysell_test(self):
        self.env.reset()

        self.env.step(np.array([0.1, 0.1]))
        (cash, quantity, price), reward, done, _ = self.env.step(np.array([-1., -1.]))

        np.testing.assert_array_almost_equal(0, quantity)


class SolowEnvTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        super(SolowEnvTests, cls).setUpClass()
        cls.static_env = fed_env.SolowSSEnv(sigma=0.)
        cls.stochastic_env = fed_env.SolowEnv(sigma=0.02)

    def steady_state_test(self):
        self.static_env.reset()
        savings = 0.1
        k_ss = (savings / self.static_env.delta) ** (1 / (1 - self.static_env.alpha))

        for _ in xrange(10000):
            state, consumption, done, _ = self.static_env.step(savings)
            capital = np.exp(state[0])

        self.assertFalse(done)
        np.testing.assert_almost_equal(capital, k_ss)

    def stochastic_state_test(self):
        self.stochastic_env.reset()
        savings = 0.1
        k_ss = (savings / self.stochastic_env.delta) ** (1 / (1 - self.stochastic_env.alpha))

        capital_states = []

        for _ in xrange(100000):
            state, consumption, done, _ = self.stochastic_env.step(savings)
            capital = np.exp(state[0])
            capital_states.append(capital)

        self.assertFalse(done)
        np.testing.assert_almost_equal(np.mean(capital_states), k_ss, decimal=1)
