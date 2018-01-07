
import numpy as np

import gym
from gym import spaces
from data import sampler


class TickerEnv(gym.Env):
    BUY_IDX = 1
    SELL_IDX = 2

    def __init__(self, starting_balance=10.):
        super(TickerEnv, self).__init__()

        self.MIN_CASH = 1.

        self.starting_balance = starting_balance

        self.cash_balance = None
        self.price = None
        self.assets = None
        self.quantity = None

        self.data = sampler.OpenCloseSampler(ticker='IEF')
        self.data_idx = None

        self.observation_space = spaces.Tuple(
            [
                spaces.Discrete(3),
                spaces.Box(0., 1, shape=1),
            ]
        )

    def _step(self, action):

        discrete_choice = action[0]
        continuous_choice = action[1]
        if discrete_choice == 0:
            q_add = 0
        else:
            if discrete_choice == 1:
                q_add = (continuous_choice * self.cash_balance / self.price)
            elif discrete_choice == 2:
                q_add = - continuous_choice * self.quantity
            else:
                raise ValueError('Valid choices are [0, 1, 2]')

        self.quantity += q_add
        self.cash_balance += -(q_add * self.price).sum()

        old_assets = self.assets
        self.assets = self.cash_balance + np.sum(self.quantity * self.price)
        done = self.assets < self.MIN_CASH

        self.data_idx += 1
        self.price = self.price_vol_data[self.data_idx, 0]
        self.volume = self.price_vol_data[self.data_idx, 1]

        return (
            np.hstack([self.cash_balance, self.quantity, self.price, self.volume]).flatten(),
            np.log(self.assets + 1e-4) - np.log(old_assets + 1e-4),
            done,
            {}
        )

    def _reset(self):
        self.cash_balance = self.starting_balance
        self.assets = self.cash_balance
        self.price_vol_data = self.data.sample(1024)

        self.data_idx = 0
        self.price = self.price_vol_data[self.data_idx, 0]
        self.volume = self.price_vol_data[self.data_idx, 1]

        self.quantity = 0.

        return np.hstack([self.cash_balance, self.quantity, self.price, self.volume])

    def _seed(self, seed=None):
        if seed:
            np.random.seed(seed)


class SolowEnv(gym.Env):
    """
    Classic Solow model (no growth or pop growth) with log consumption utility
    States are histories of capital and tech innovation/shock
    """
    def __init__(self, delta=0.02, sigma=0.1):
        super(SolowEnv, self).__init__()

        self.delta = delta
        self.sigma = sigma
        self.rho = 0.95
        self.alpha = 0.33

        self.z = None
        self.k = None

        self.action_space = spaces.Box(0, 1., shape=1)

    def _k_transition(self, k_t, y_t, s):
        return (1 - self.delta) * k_t + s * y_t

    def _k_ss(self, savings):
        return (savings / self.delta) ** (1 / (1 - self.alpha))

    def _step(self, s):
        y_t = np.exp(self.z) * (self.k ** self.alpha)
        z_next = self.rho * self.z + np.random.normal(0, self.sigma)
        k_next = self._k_transition(self.k, y_t, s)

        self.z = z_next
        self.k = k_next

        state = np.array([self.k, self.z]).flatten()

        return (
            state,
            (1 - s) * y_t,
            s <= 0 or s >= 1.,
            {}
        )

    def _reset(self):
        self.k = self._k_ss(np.random.uniform(0.05, 0.9))
        self.z = np.random.uniform(-1e-2, 1e-2)

        return np.array([self.k, self.z]).flatten()


class SolowSSEnv(SolowEnv):
    def __init__(self, delta=0.02, sigma=0.02):
        super(SolowSSEnv, self).__init__(delta, sigma)

    def _reset(self):
        self.k = self._k_ss(self.alpha)
        self.z = np.random.uniform(-1e-2, 1e-2)

        return np.array([self.k, self.z]).flatten()


class TradeAR1Env(gym.Env):
    def __init__(self, starting_balance=10., base_rate=0.05, n_assets=2, std_p=0.05):
        super(TradeAR1Env, self).__init__()

        self.MIN_CASH = 1.

        self.starting_balance = starting_balance
        self.r = base_rate
        self.n_assets = n_assets
        self.rho_p = 0.9
        self.std_e = np.sqrt((std_p ** 2) * (1 - self.rho_p ** 2))

        self.cash_balance = None
        self.prices = None
        self.assets = None
        self.quantity = None
        self.e = None

        # fraction to sell = negative, fraction of funds used to purchase = positive
        self.action_space = spaces.Box(-1., 1., shape=(self.n_assets, ))
        self.observation_space = spaces.Tuple(
            [
                spaces.Box(0., np.inf, shape=(1, )), # funds
                spaces.Box(0., np.inf, shape=(2, )), # quantity
                spaces.Box(0., np.inf, shape=(n_assets, )) # price
            ]
        )

    def _price_transition(self, p):
        e_t = self.std_e * np.random.normal(size=(self.n_assets, ))
        return (p ** self.rho_p) * np.exp(e_t)

    def _step(self, action):
        assert self.action_space.contains(action)
        buy_mask = action > 0
        q_add = np.zeros_like(action)
        q_add[buy_mask] = ((action / self.n_assets) * self.cash_balance / self.prices)[buy_mask]
        q_add[~buy_mask] = (action * self.quantity)[~buy_mask]

        self.quantity += q_add
        self.cash_balance += -(q_add * self.prices).sum()

        old_assets = self.assets
        self.assets = self.cash_balance + np.sum(self.quantity * self.prices)
        done = self.assets < self.MIN_CASH

        self.prices = self._price_transition(self.prices)

        return (
            np.hstack([self.cash_balance, self.quantity, self.prices]).flatten(),
            np.log(self.assets + 1e-4) - np.log(old_assets + 1e-4),
            done,
            {}
        )

    def _reset(self):
        self.cash_balance = self.starting_balance
        self.assets = self.cash_balance
        self.prices = np.ones((self.n_assets, ))
        self.quantity = np.zeros((self.n_assets, ))
        self.e = np.zeros_like(self.quantity)

        return np.hstack([self.cash_balance, self.quantity, self.prices])

    def _seed(self, seed=None):
        if seed:
            np.random.seed(seed)
