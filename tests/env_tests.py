import unittest

import numpy as np

from fed_gym.envs import fed_env, multiagent


class SwarmTests(unittest.TestCase):
    def run_env_test(self):
        def fake_action():
            Na = env.N_AGENTS
            va = np.zeros((Na, 2))
            for i in range(Na):
                va[i, :] = np.random.normal(size=(2, ))
            return va

        env = multiagent.SwarmEnv()
        env.reset()
        for _ in range(20):
            action = fake_action()
            state, reward, done, _ = env.step(action)

        self.assertEqual(len(state), 2)
        self.assertEqual(state[0].shape, (env.N_LOCUSTS, 2))
        self.assertEqual(state[1].shape, (env.N_AGENTS, 2))
        self.assertLess(reward, 0.)
        self.assertFalse(done)


class TickerEnvTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        super(TickerEnvTests, cls).setUpClass()
        cls.n_assets = 7
        cls.env = fed_env.TickerEnvForTests(n_assets=cls.n_assets)

    def deplete_test(self):
        self.env.reset()

        for _ in range(100):
            state, reward, done, _ = self.env.step(
                [np.array([self.env.BUY_IDX] * self.n_assets), np.array([0.1] * self.n_assets)]
            )

        cash = state[0]
        quantity = state[1: 1 + self.n_assets]

        self.assertEqual(done, False)
        self.assertLessEqual(cash, self.env.MIN_CASH)
        np.testing.assert_array_less(0, quantity)

    def buysell_test(self):
        self.env.reset()

        self.env.step(
            [np.array([self.env.BUY_IDX] * self.n_assets), np.array([0.1] * self.n_assets)]
        )
        state, reward, done, _ = self.env.step(
            [np.array([self.env.SELL_IDX] * self.n_assets), np.array([1.] * self.n_assets)]
        )

        quantity = state[1: 1 + self.n_assets]
        np.testing.assert_array_almost_equal(0, quantity)


class TradingEnvTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        super(TradingEnvTests, cls).setUpClass()
        cls.STD_P = 0.05
        cls.env = fed_env.TradeAR1Env(std_p=cls.STD_P)

    def deplete_test(self):
        self.env.reset()

        for _ in range(100):
            state, reward, done, _ = self.env.step(np.array([0.1, 0.1]))

        cash = state[0]
        quantity = state[1:3]

        self.assertEqual(done, False)
        self.assertLessEqual(cash, self.env.MIN_CASH)
        np.testing.assert_array_less(0, quantity)

    def buysell_test(self):
        self.env.reset()

        self.env.step(np.array([0.1, 0.1]))
        state, reward, done, _ = self.env.step(np.array([-1., -1.]))

        quantity = state[1:3]

        np.testing.assert_array_almost_equal(0, quantity)

    def prices_test(self):
        self.env.reset()

        p = []
        for _ in range(100):
            state, reward, done, _ = self.env.step(np.array([0.0, 0.0]))
            prices = state[3:5]
            p.append(prices)

        np.testing.assert_array_less(np.std(p, axis=0), self.STD_P * 2)


class SolowEnvTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        super(SolowEnvTests, cls).setUpClass()
        cls.static_env = fed_env.SolowSSEnv(sigma=0., T=10000)
        cls.stochastic_env = fed_env.SolowEnv(sigma=0.02, T=100000)
        cls.arima_env = fed_env.SolowEnv(p=3, q=2)

    def arima_test(self):
        self.arima_env.reset()
        savings = 0.1

        for _ in range(100):
            state, consumption, done, _ = self.arima_env.step(savings)

        self.assertFalse(done)

    def steady_state_test(self):
        self.static_env.reset()
        savings = 0.1
        k_ss = (savings / self.static_env.delta) ** (1 / (1 - self.static_env.alpha))

        for _ in range(10000):
            state, consumption, done, _ = self.static_env.step(savings)
            capital = state[0]

        self.assertFalse(done)
        np.testing.assert_almost_equal(capital, k_ss)

    def stochastic_state_test(self):
        self.stochastic_env.reset()
        savings = 0.1
        k_ss = (savings / self.stochastic_env.delta) ** (1 / (1 - self.stochastic_env.alpha))

        capital_states = []

        for _ in range(100000):
            state, consumption, done, _ = self.stochastic_env.step(savings)
            capital = state[0]
            capital_states.append(capital)

        self.assertFalse(done)
        np.testing.assert_almost_equal(np.mean(capital_states), k_ss, decimal=1)
