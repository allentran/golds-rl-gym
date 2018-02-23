import numpy as np


class StateProcessor(object):
    def __init__(self, scales):
        self.scales = scales

    def process_temporal_states(self, history):
        raise NotImplementedError

    def process_state(self, state):
        return state / self.scales


class SwarmStateProcessor(StateProcessor):

    def __init__(self, scales=1., grid_size=40):
        super().__init__(scales)
        self.grid_size = grid_size
        self.positions = None

    def hist_calc(self, x, update_position=False):
        N = x.shape[0]
        u = np.zeros((self.grid_size, self.grid_size))
        amin = np.amin(x, axis=0)
        amax = np.amax(x, axis=0)
        xmin0 = amin[0]
        xmax0 = amax[0] + 0.000001
        xmin1 = amin[1]
        xmax1 = amax[1] + 0.000001

        if update_position:
            self.positions = np.zeros((N, 2), dtype='int32')

        for j in range(N):
            xs=np.int(np.floor(self.grid_size *(x[j][0] - xmin0) / (xmax0 - xmin0)))
            ys=np.int(np.floor(self.grid_size *(x[j][1] - xmin1) / (xmax1 - xmin1)))

            u[xs,ys] += 1
            if update_position:
                self.positions[j] = [xs, ys]

        return u

    def process_state(self, state):
        grid = np.stack([self.hist_calc(state[0]), self.hist_calc(state[1], update_position=True)], axis=-1)
        return grid


class TickerTraderStateProcessor(StateProcessor):
    def __init__(self, n_assets):
        super(TickerTraderStateProcessor, self).__init__(None)
        self.n_assets = n_assets

    def process_state(self, raw_state):
        cash = raw_state[0]
        quantities = raw_state[1: 1 + self.n_assets]
        prices = raw_state[1 + self.n_assets: -self.n_assets]
        volumes = raw_state[-self.n_assets:]

        state = [np.log(cash + 1e-4)]
        for idx in range(self.n_assets):
            state.append(np.log(quantities[idx] + 1))
        for idx in range(self.n_assets):
            state.append(np.log(prices[idx]))
        for idx in range(self.n_assets):
            state.append(volumes[idx])
        return np.array(state).flatten()

    def process_temporal_states(self, history):
        return np.vstack(history)[:, 1 + self.n_assets:]


class SolowStateProcessor(StateProcessor):
    def __init__(self):
        super(SolowStateProcessor, self).__init__(np.array([100., 1.]))

    def process_temporal_states(self, history):
        if len(history) == 1:
            return np.array(history[0]).reshape((1, -1))
        return np.array(history)