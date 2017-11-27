
import numpy as np

import gym
from gym import error, spaces, utils
from gym.utils import seeding


class FedEnv(gym.Env):
    def __init__(self, starting_balance=100., base_rate=0.05):
        super(FedEnv, self).__init__()

        self.starting_balance = starting_balance
        self.r = base_rate

        self.cash_balance = None
        self.price = None
        self.quantity = None
        self.e = None

        # fraction to sell = negative, fraction of funds used to purchase = positive
        self.action_space = spaces.Box(-1., 1.)
        self.observation_space = spaces.Tuple(
            [
                spaces.Box(0., 10e5), # funds
                spaces.Box(0., 10e5), # quantity
                spaces.Box(0., 1.) # price
            ]
        )

    def _price_transition(self, p):
        rho = 0.9
        std_e = 0.1
        self.e = rho * self.e + np.random.normal(std_e)
        return p * np.exp(self.e)

    def _step(self, action):
        assert self.action_space.contains(action)
        if action > 0:
            q_add = action * self.cash_balance / self.price
        else:
            q_add = - action * self.quantity

        reward = self.cash_balance * self.r

        self.quantity += q_add
        self.cash_balance += -q_add * self.price
        self.price = self._price_transition(self.price)

        return (
            [self.cash_balance, self.quantity, self.price],
            reward,
            self.cash_balance <= 0,
            {}
        )

    def _reset(self):
        self.cash_balance = self.starting_balance
        self.price = np.random.uniform(5, 10)
        self.quantity = 0
        self.e = 0.

    def _seed(self, seed=None):
        if seed:
            np.random.seed(seed)

    def _render(self, mode='human', close=False):
        super(FedEnv, self)._render(mode, close)




