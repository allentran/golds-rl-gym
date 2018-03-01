from multiprocessing import Process

import tensorflow as tf
import numpy as np

from ..a3c.worker import sigmoid
from fed_gym.agents.state_processors import SolowStateProcessor, SwarmStateProcessor


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
        self.state_processor = None

        self.rnn_length = self.variables[self.HISTORY_IDX].shape[1]

    @staticmethod
    def transform_actions_for_env(actions):
        return actions

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


class SolowRunner(EmulatorRunner):

    def __init__(self, id, emulators, variables, queue, barrier):
        super().__init__(id, emulators, variables, queue, barrier)
        self.state_processor = SolowStateProcessor()

    @staticmethod
    def transform_actions_for_env(actions):
        return sigmoid(actions)


class SwarmRunner(EmulatorRunner):

    STATE_IDX = 0
    HISTORY_IDX = 1
    AGENT_POSITIONS_IDX = 2
    REWARD_IDX = 3
    DONE_IDX = 4
    ACTIONS_IDX = 5

    MAX_MOVE_NORM = 1

    def __init__(self, id, emulators, variables, queue, barrier):
        super().__init__(id, emulators, variables, queue, barrier)
        self.rnn_length = self.variables[self.HISTORY_IDX].shape[2]
        self.state_processor = SwarmStateProcessor()

    @staticmethod
    def get_local_states(state, agent_positions):
        """
        Get spatially local state
        :param state:
        :param agent_positions:
        :return:
        """
        # TODO: add noise or black out states
        return [state] * len(agent_positions)

    @staticmethod
    def transform_actions_for_env(actions):
        distance = np.linalg.norm(actions, axis=-1)
        move_too_much_mask = distance >= SwarmRunner.MAX_MOVE_NORM
        actions[move_too_much_mask, :] /= distance[move_too_much_mask][:, None]
        return actions

    def _run(self):
        count = 0
        while True:
            instruction = self.queue.get()
            if instruction is None:
                break
            for i, (emulator, action) in enumerate(zip(self.emulators, self.variables[-1])):
                new_s, reward, episode_over, _ = emulator.step(action)
                if episode_over:
                    processed_state = self.state_processor.process_state(emulator.reset())
                    local_states = self.get_local_states(processed_state, self.state_processor.positions)
                    self.variables[self.STATE_IDX][i] = local_states
                    self.histories[i] = [self.variables[self.STATE_IDX][i]]
                else:
                    processed_state = self.state_processor.process_state(new_s)
                    local_states = self.get_local_states(processed_state, self.state_processor.positions)
                    self.variables[self.STATE_IDX][i] = local_states
                    self.histories[i].append(self.variables[self.STATE_IDX][i])

                histories = np.array(self.histories[i])[-self.rnn_length:]
                histories = np.swapaxes(histories, 0, 1)
                if len(self.histories[i]) < self.rnn_length:
                    histories = tf.keras.preprocessing.sequence.pad_sequences(histories, maxlen=self.rnn_length, padding='post')
                self.histories[i] = self.histories[i][-(self.rnn_length + 1):]

                self.variables[self.AGENT_POSITIONS_IDX][i] = self.state_processor.positions
                self.variables[self.HISTORY_IDX][i] = histories
                self.variables[self.REWARD_IDX][i] = reward
                self.variables[self.DONE_IDX][i] = episode_over

            count += 1
            self.barrier.put(True)
