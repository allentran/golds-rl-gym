from multiprocessing import Process

import numpy as np

from fed_gym.agents.a3c.estimators import SolowStateProcessor


class EmulatorRunner(Process):

    STATE_IDX = 0
    HISTORY_IDX = 1
    REWARD_IDX = 2
    DONE_IDX = 3
    ACTIONS_IDX = 4

    def __init__(self, id, emulators, variables, queue, barrier):
        super(EmulatorRunner, self).__init__()
        self.id = id
        self.emulators = emulators
        self.variables = variables
        self.histories = [[] for _ in range(len(emulators))]
        self.queue = queue
        self.barrier = barrier
        self.state_processor = SolowStateProcessor()

        self.rnn_length = self.variables[self.HISTORY_IDX].shape[1]

    def run(self):
        super(EmulatorRunner, self).run()
        self._run()

    def _run(self):
        """
        Runs each emulator/env one-step for the actions in self.variables[-1]
        :return:
        """
        count = 0
        while True:
            instruction = self.queue.get()
            if instruction is None:
                break
            for i, (emulator, action) in enumerate(zip(self.emulators, self.variables[-1])):
                new_s, reward, episode_over, _ = emulator.step(action)
                if episode_over:
                    self.variables[self.STATE_IDX][i] = self.state_processor.process_state(emulator.reset())
                    self.histories[i] = [self.variables[self.STATE_IDX][i]]
                else:
                    self.variables[self.STATE_IDX][i] = self.state_processor.process_state(new_s)
                    self.histories[i].append(self.variables[self.STATE_IDX][i])

                histories = np.vstack(self.histories[i])[-self.rnn_length:]
                if len(histories) < self.rnn_length:
                    n_pad = self.rnn_length - len(histories)
                    histories = np.concatenate((histories, np.zeros((n_pad, histories.shape[1]))))
                self.histories[i] = self.histories[i][-(self.rnn_length + 1):]

                self.variables[self.HISTORY_IDX][i] = histories
                self.variables[self.REWARD_IDX][i] = reward
                self.variables[self.DONE_IDX][i] = episode_over

            count += 1
            self.barrier.put(True)
