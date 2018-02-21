import numpy as np


class StateProcessor(object):
    def __init__(self, scales):
        self.scales = scales

    def process_temporal_states(self, history):
        raise NotImplementedError

    def process_state(self, state):
        return state / self.scales


class SwarmStateProcessor(StateProcessor):

    def __init__(self, scales=1.):
        super().__init__(scales)

    def process_state(self, state):
        print(state)
        return super().process_state(state)


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