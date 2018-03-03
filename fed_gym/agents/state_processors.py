import numpy as np


class StateProcessor(object):
    def __init__(self, scales):
        self.scales = scales

    def process_temporal_states(self, history):
        raise NotImplementedError

    def process_state(self, state):
        return state / self.scales


class SwarmStateProcessor(StateProcessor):

    def __init__(self, scales=1., grid_size=20):
        super().__init__(scales)
        self.grid_size = grid_size
        self.positions = None

        self.WIDTH = 3.
        self.HEIGHT = 3.

    def _get_bounding_box(self, x):
        mean_x = np.median(x, axis=0)[0]
        return [[mean_x - self.WIDTH / 2., mean_x + self.WIDTH / 2.], [0, self.HEIGHT]]

    def process_state(self, state):
        bounding_box = self._get_bounding_box(state[0])
        x_grid, x_edges, y_edges = np.histogram2d(state[0][:, 0], state[0][:, 1], self.grid_size, bounding_box)
        xa_grid, xa_x_edges, xa_y_edges = np.histogram2d(state[1][:, 0], state[1][:, 1], bins=[x_edges, y_edges])
        grid = np.stack([x_grid, xa_grid], axis=-1)

        xa_x_idx = np.digitize(state[1][:, 0], x_edges)
        xa_x_idx[xa_x_idx >= self.grid_size] = self.grid_size - 1
        xa_y_idx = np.digitize(state[1][:, 1], y_edges)
        xa_y_idx[xa_y_idx >= self.grid_size] = self.grid_size - 1

        self.positions = np.hstack([xa_x_idx[:, None], xa_y_idx[:, None]]).astype('uint8')

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