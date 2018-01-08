import os
import random

import numpy as np
import pandas as pd


class OpenCloseSampler(object):
    def __init__(self, ticker):
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '%s.csv' % ticker)
        raw_data = pd.read_csv(path)
        self.data_matrix = self.open_close_to_sequence(raw_data)
        self.T = len(self.data_matrix)

    def open_close_to_sequence(self, open_close_data, open_col='Open', close_col='Close', vol_col='Volume'):
        opens = open_close_data[open_col].values[:, None]
        closes = open_close_data[close_col].values[:, None]
        vol = open_close_data[vol_col][:, None]

        joined = np.hstack([opens, closes]).flatten()
        vol = np.hstack([vol, vol]).flatten()
        vol = np.log(vol) - np.log(vol[0])
        data = np.hstack([joined[:, None], vol[:, None]])

        assert data[1, 0] == open_close_data[close_col].iloc[0]

        return data

    def sample(self, n):
        start_idx = random.randint(0, self.T - n)
        return self.data_matrix[start_idx: start_idx + n]


