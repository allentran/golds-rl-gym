import os
import random

import numpy as np
import pandas as pd


class OpenCloseSampler(object):
    def __init__(self, ticker, inverse_asset=True):
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '%s.csv' % ticker)
        raw_data = pd.read_csv(path)
        self.data_matrix = self.open_close_to_sequence(raw_data, inverse_asset=inverse_asset)
        self.T = len(self.data_matrix)

    def open_close_to_sequence(self, open_close_data, open_col='Open', close_col='Close', vol_col='Volume', inverse_asset=True):
        opens = open_close_data[open_col].values[:, None]
        closes = open_close_data[close_col].values[:, None]
        vol = open_close_data[vol_col][:, None]

        joined_prices = np.hstack([opens, closes]).flatten()
        inverse_joined = self._get_inverse(joined_prices)
        vol = np.hstack([vol, vol]).flatten()
        vol = np.log(vol) - np.log(vol[0])
        data = np.hstack([joined_prices[:, None], inverse_joined[:, None], vol[:, None], vol[:, None]])

        assert data[1, 0] == open_close_data[close_col].iloc[0]

        return data

    def _get_inverse(self, prices):
        returns = np.log(prices[1:]) - np.log(prices[0: -1])
        cum_neg_returns = [0] + np.cumsum(- returns).tolist()
        inverse_prices = prices[0] * np.exp(cum_neg_returns)
        d_inv = np.log(inverse_prices[1:]) - np.log(inverse_prices[0: -1])
        d = np.log(prices[1:]) - np.log(prices[0: -1])
        assert np.abs(-d_inv - d).sum() == 0
        return inverse_prices

    def sample(self, n):
        start_idx = random.randint(0, self.T - n)
        return self.data_matrix[start_idx: start_idx + n]


if __name__ == "__main__":
    o = OpenCloseSampler('IEF', True)
