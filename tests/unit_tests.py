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
            (cash, quantity, price), reward, done, _ = self.env.step(np.array([0.1, 0.1]))

        self.assertTrue(done)
        self.assertLessEqual(cash, self.env.MIN_CASH)
        np.testing.assert_array_less(0, quantity)
        np.testing.assert_array_less(0, price)

    def buysell_test(self):
        self.env.reset()

        self.env.step(np.array([0.1, 0.1]))
        (cash, quantity, price), reward, done, _ = self.env.step(np.array([-1., -1.]))

        np.testing.assert_array_almost_equal(0, quantity)
