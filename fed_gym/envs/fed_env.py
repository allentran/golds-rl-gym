
import numpy as np

import gym
from gym import error, spaces, utils
from gym.utils import seeding


class FedEnv(gym.Env):
    def __init__(self, starting_balance=100., base_rate=0.05, n_assets=2):
        super(FedEnv, self).__init__()

        self.MIN_CASH = 10.

        self.starting_balance = starting_balance
        self.r = base_rate
        self.n_assets = n_assets
        self.cov_mat = self._get_cov_mat()

        self.cash_balance = None
        self.price = None
        self.quantity = None
        self.e = None

        # fraction to sell = negative, fraction of funds used to purchase = positive
        self.action_space = spaces.Box(-1., 1., shape=(self.n_assets, ))
        self.observation_space = spaces.Tuple(
            [
                spaces.Box(0., 10e5, shape=(1, )), # funds
                spaces.Box(0., 10e5, shape=(2, )), # quantity
                spaces.Box(0., 1., shape=(2, )) # price
            ]
        )

    def _get_cov_mat(self):
        std_e = 1e-3
        cov = np.zeros((self.n_assets, self.n_assets))
        np.fill_diagonal(cov, std_e)

        return cov

    def _price_transition(self, p):
        rho = 0.9
        self.e = rho * self.e + np.random.multivariate_normal(
            np.zeros((self.n_assets, )), self.cov_mat
        )
        return p * np.exp(self.e)

    def _step(self, action):
        assert self.action_space.contains(action)
        buy_mask = action > 0
        q_add = np.zeros_like(action)
        q_add[buy_mask] = (action * self.cash_balance / self.price)[buy_mask]
        q_add[~buy_mask] = (action * self.quantity)[~buy_mask]

        reward = self.cash_balance * self.r

        self.quantity += q_add
        self.cash_balance += -(q_add * self.price).sum()
        self.price = self._price_transition(self.price)

        return (
            [self.cash_balance, self.quantity, self.price],
            reward,
            self.cash_balance <= self.MIN_CASH,
            {}
        )

    def _reset(self):
        self.cash_balance = self.starting_balance
        self.price = np.random.uniform(5, 10, size=(self.n_assets, ))
        self.quantity = np.zeros((self.n_assets, ))
        self.e = np.zeros_like(self.quantity)

        return [self.cash_balance, self.quantity, self.price]

    def _seed(self, seed=None):
        if seed:
            np.random.seed(seed)

    def _render(self, mode='human', close=False):
        super(FedEnv, self)._render(mode, close)




