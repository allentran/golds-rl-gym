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

    def hist_calc(self, x, update_position=False, min_x=None, max_x=None, min_y=None, max_y=None):
        N = x.shape[0]
        u = np.zeros((self.grid_size, self.grid_size))

        if update_position:
            self.positions = np.zeros((N, 2), dtype='int32')

        frac_x = (x[:, 0] - min_x) / (max_x - min_x)
        frac_y = (x[:, 1] - min_y) / (max_y - min_y)

        xs = np.floor(self.grid_size * frac_x).astype('int32')
        ys = np.floor(self.grid_size * frac_y).astype('int32')

        if update_position:
            self.positions = np.hstack([xs[:, None], ys[:, None]])

        for j in range(N):
            x, y = xs[j], ys[j]
            u[x, y] += 1

        return u

    @staticmethod
    def get_bounding_box(position_arrays):
        x_min, x_max, y_min, y_max = np.inf, -np.inf, np.inf, -np.inf
        array = np.vstack(position_arrays)
        a_xmin, a_ymin = array.min(axis=0)
        a_xmax, a_ymax = array.max(axis=0)
        if a_xmin < x_min:
            x_min = a_xmin
        if a_xmax > x_max:
            x_max = a_xmax
        if a_ymin < y_min:
            y_min = a_ymin
        if a_ymax > y_max:
            y_max = a_ymax
        return x_min, x_max + 1e-6, y_min, y_max + 1e-6

    def process_state(self, state):
        bounding_box = self.get_bounding_box(state)
        grid = np.stack(
            [self.hist_calc(state[0], False, *bounding_box), self.hist_calc(state[1], True, *bounding_box)],
            axis=-1
        )
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