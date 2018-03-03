import time
import threading
from multiprocessing.sharedctypes import RawArray
from ctypes import c_float
import logging

import gym
import numpy as np

from .actor_learner import *
from .runners import Runners
from ..state_processors import SwarmStateProcessor, SolowStateProcessor
from .policy_monitor import SolowPolicyMonitor, SwarmPolicyMonitor


class PAACLearner(ActorLearner):
    def __init__(self, network_creator, environment_creator, args, emulator_class, state_processor):
        super(PAACLearner, self).__init__(network_creator, environment_creator, args, emulator_class)
        self.workers = args.emulator_workers
        self.rnn_length = args.rnn_length
        self.state_processor = state_processor

    @staticmethod
    def choose_next_actions(network, num_actions, states, histories, session):
        network_output_mu, network_output_sigma, network_output_v = session.run(
            [
                network.mu,
                network.sigma,
                network.vs
            ],
            feed_dict={
                network.states: states,
                network.history: histories
            }
        )
        new_actions = network_output_mu + network_output_sigma * np.random.normal(size=network_output_mu.shape)

        return new_actions, network_output_v

    def _choose_next_actions(self, states, histories):
        return self.choose_next_actions(self.network, self.num_actions, states, histories, self.session)

    def _get_shared(self, array, dtype=c_float):
        """
        Returns a RawArray backed numpy array that can be shared between processes.
        :param array: the array to be shared
        :param dtype: the RawArray dtype to use
        :return: the RawArray backed numpy array
        """

        shape = array.shape
        shared = RawArray(dtype, array.reshape(-1))
        return np.frombuffer(shared, dtype).reshape(shape)

    def train(self):
        """
        Main actor learner loop for parallel advantage actor critic learning.
        """

        self.global_step = self.init_network()

        coord = tf.train.Coordinator()
        pe = SolowPolicyMonitor(
            env=gym.envs.make("Solow-1-1-finite-eval-v0"),
            global_policy_net=self.network,
            state_processor=SolowStateProcessor(),
            summary_writer=self.summary_writer,
            saver=None,
            network_conf=self.network.conf,
        )

        monitor_thread = threading.Thread(
            target=lambda: pe.continuous_eval(
                10., self.session, coord, self.rnn_length
            )
        )
        monitor_thread.start()

        logging.debug("Starting training at Step {}".format(self.global_step))
        counter = 0

        global_step_start = self.global_step

        total_rewards = []

        # state, histories, reward, episode_over, action
        initial_states = [self.state_processor.process_state(emulator.reset()) for emulator in self.emulators]
        temporal_state_matrix = tf.keras.preprocessing.sequence.pad_sequences(
            [[x] for x in initial_states], dtype='float32', padding='post', maxlen=self.rnn_length
        )
        variables = [
            np.array(initial_states),
            temporal_state_matrix,
            (np.zeros(self.emulator_counts, dtype=np.float32)),
            (np.asarray([False] * self.emulator_counts, dtype=np.float32)),
            (np.zeros((self.emulator_counts, self.num_actions), dtype=np.float32))
        ]

        self.runners = Runners(self.emulators, self.workers, variables, self.emulator_class, coord)
        self.runners.start()
        shared_states, shared_histories, shared_rewards, shared_episode_over, shared_actions = self.runners.get_shared_variables()

        summaries_op = tf.summary.merge_all()

        emulator_steps = [0] * self.emulator_counts
        total_episode_rewards = self.emulator_counts * [0]

        y_batch = np.zeros((self.max_local_steps, self.emulator_counts))
        adv_batch = np.zeros((self.max_local_steps, self.emulator_counts))
        rewards = np.zeros((self.max_local_steps, self.emulator_counts))
        states = np.zeros([self.max_local_steps] + list(shared_states.shape))
        actions = np.zeros((self.max_local_steps, self.emulator_counts, self.num_actions))
        histories = np.zeros((self.max_local_steps, self.emulator_counts, self.rnn_length, shared_histories.shape[-1]))
        values = np.zeros((self.max_local_steps, self.emulator_counts))
        episodes_over_masks = np.zeros((self.max_local_steps, self.emulator_counts))

        start_time = time.time()

        while self.global_step < self.max_global_steps:

            loop_start_time = time.time()

            max_local_steps = self.max_local_steps
            for t in range(max_local_steps):
                next_actions, readouts_v_t = self._choose_next_actions(shared_states, shared_histories)
                transformed_actions = self.emulator_class.transform_actions_for_env(next_actions)
                for z in range(next_actions.shape[0]):
                    shared_actions[z] = transformed_actions[z]

                actions[t] = next_actions
                values[t] = readouts_v_t
                states[t] = shared_states
                histories[t] = shared_histories

                # Start updating all environments with next_actions
                self.runners.update_environments()
                self.runners.wait_updated()
                # Done updating all environments, have new states, rewards and is_over

                episodes_over_masks[t] = 1.0 - shared_episode_over.astype(np.float32)

                for e_idx, (actual_reward, episode_over) in enumerate(zip(shared_rewards, shared_episode_over)):

                    total_episode_rewards[e_idx] += actual_reward
                    actual_reward = self.rescale_reward(actual_reward)
                    rewards[t, e_idx] = actual_reward

                    emulator_steps[e_idx] += 1
                    self.global_step += 1
                    if episode_over:
                        total_rewards.append(total_episode_rewards[e_idx] / emulator_steps[e_idx])
                        episode_summary = tf.Summary()
                        episode_summary.value.add(simple_value=total_episode_rewards[e_idx], tag="rl/reward")
                        self.summary_writer.add_summary(episode_summary, self.global_step)
                        self.summary_writer.flush()
                        total_episode_rewards[e_idx] = 0
                        emulator_steps[e_idx] = 0

            next_state_value = self.session.run(
                self.network.vs,
                feed_dict={
                    self.network.states: shared_states,
                    self.network.history: shared_histories
                }
            )

            estimated_return = np.copy(next_state_value)

            for t in reversed(range(max_local_steps)):
                estimated_return = rewards[t] + self.gamma * estimated_return * episodes_over_masks[t]
                y_batch[t] = np.copy(estimated_return)
                adv_batch[t] = estimated_return - values[t]

            flat_states = states.reshape([self.max_local_steps * self.emulator_counts] + list(shared_states.shape)[1:])
            flat_history = histories.reshape((self.max_local_steps * self.emulator_counts, self.rnn_length, shared_histories.shape[-1]))
            flat_y_batch = y_batch.reshape(-1)
            flat_adv_batch = adv_batch.reshape(-1) / self.network.scale
            flat_actions = actions.reshape(max_local_steps * self.emulator_counts, self.num_actions)
            lr = self.get_lr()
            feed_dict = {
                self.network.states: flat_states,
                self.network.history: flat_history,
                self.network.critic_target: flat_y_batch,
                self.network.actions: flat_actions,
                self.network.advantages: flat_adv_batch,
                self.learning_rate: lr
            }

            _, summaries, global_step = self.session.run(
                [self.train_step, summaries_op, self.network.global_step_tensor],
                feed_dict=feed_dict)

            self.summary_writer.add_summary(summaries, global_step)
            self.summary_writer.flush()

            counter += 1

            if counter % (5048 / self.emulator_counts) == 0:
                curr_time = time.time()
                global_steps = self.global_step
                last_ten = 0.0 if len(total_rewards) < 1 else np.mean(total_rewards[-10:])
                logging.info("Ran {} steps, at {} steps/s ({} steps/s avg), last 10 rewards avg {}"
                             .format(global_steps,
                                     self.max_local_steps * self.emulator_counts / (curr_time - loop_start_time),
                                     (global_steps - global_step_start) / (curr_time - start_time),
                                     last_ten))
            self.save_vars()

        self.cleanup()

    def cleanup(self):
        super(PAACLearner, self).cleanup()
        self.runners.stop()


class GridPAACLearner(PAACLearner):
    N_AGENTS = 10

    def __init__(self, network_creator, environment_creator, args, emulator_class, state_processor):
        super().__init__(network_creator, environment_creator, args, emulator_class, state_processor)
        self.real_batch_size = self.emulator_counts * self.N_AGENTS

    def rescale_reward(self, reward, lb=-2, ub=2):
        return reward

    def train(self):

        self.global_step = self.init_network()

        coord = tf.train.Coordinator()
        pe = SwarmPolicyMonitor(
            env=gym.envs.make("Swarm-eval-v0"),
            global_policy_net=self.network,
            state_processor=SwarmStateProcessor(grid_size=self.network.height),
            summary_writer=self.summary_writer,
            saver=None,
            network_conf=self.network.conf,
        )

        logging.debug("Starting training at Step {}".format(self.global_step))

        total_rewards = []

        # state, histories, reward, episode_over, action
        state_idxs = []
        initial_states = []
        for emulator in self.emulators:
            initial_state = self.state_processor.process_state(emulator.reset())
            agent_positions = self.state_processor.positions
            initial_states.append(self.emulator_class.get_local_states(initial_state, agent_positions))
            state_idxs.append(self.state_processor.positions)

        initial_states = np.array(initial_states)

        temporal_state_matrix = initial_states.reshape(-1, *initial_states.shape[-3:])
        temporal_state_matrix = np.expand_dims(temporal_state_matrix, 1)
        temporal_state_matrix = tf.keras.preprocessing.sequence.pad_sequences(
            temporal_state_matrix, dtype='float32', padding='post', maxlen=self.rnn_length
        )
        temporal_state_matrix = np.reshape(
            temporal_state_matrix, (self.emulator_counts, self.N_AGENTS, self.rnn_length, *temporal_state_matrix.shape[-3:])
        )
        variables = [
            initial_states,
            temporal_state_matrix,
            np.array(state_idxs),
            np.zeros((self.emulator_counts, self.N_AGENTS), dtype=np.float32),
            np.asarray([False] * self.real_batch_size, dtype=np.float32).reshape((self.emulator_counts, self.N_AGENTS)),
            np.zeros((self.emulator_counts, self.N_AGENTS, self.num_actions), dtype=np.float32),
        ]

        self.runners = Runners(self.emulators, self.workers, variables, self.emulator_class, coord)
        self.runners.start()
        shared_states, shared_histories, shared_positions, shared_rewards, shared_episode_over, shared_actions = self.runners.get_shared_variables()

        summaries_op = tf.summary.merge_all()
        monitor_thread = threading.Thread(
            target=lambda: pe.continuous_eval(
                30., self.session, coord, self.rnn_length
            )
        )
        monitor_thread.start()

        emulator_steps = [0] * self.emulator_counts
        total_episode_rewards = self.emulator_counts * [0]

        y_batch = np.zeros((self.max_local_steps, self.real_batch_size))
        adv_batch = np.zeros((self.max_local_steps, self.real_batch_size))
        rewards = np.zeros((self.max_local_steps, self.real_batch_size))
        states = np.zeros([self.max_local_steps, self.real_batch_size, self.state_processor.grid_size, self.state_processor.grid_size, 2])
        actions = np.zeros((self.max_local_steps, self.real_batch_size, self.num_actions))
        positions = np.zeros((self.max_local_steps, self.real_batch_size, 2))
        histories = np.zeros((self.max_local_steps, self.real_batch_size, self.rnn_length, *states.shape[-3:]))
        values = np.zeros((self.max_local_steps, self.real_batch_size))
        episodes_over_masks = np.zeros((self.max_local_steps, self.real_batch_size))

        counter = 0
        global_step_start = self.global_step

        start_time = time.time()

        while self.global_step < self.max_global_steps:

            loop_start_time = time.time()

            max_local_steps = self.max_local_steps
            for t in range(max_local_steps):
                next_actions, readouts_v_t = self._choose_next_actions(shared_states, shared_histories, shared_positions)
                transformed_actions = self.emulator_class.transform_actions_for_env(next_actions)
                transformed_actions = transformed_actions.reshape(self.emulator_counts, self.N_AGENTS, self.num_actions)

                # NEED TO DO THIS FOR SHARED CTYPES ASSIGNMENT (IE DO NOT DO SHARED_ACTIONS = xyz)
                for idx in range(transformed_actions.shape[0]):
                    shared_actions[idx] = transformed_actions[idx]

                actions[t] = next_actions
                positions[t] = shared_positions.reshape((self.real_batch_size, 2))
                values[t] = readouts_v_t
                states[t] = shared_states.reshape((self.real_batch_size, ) + shared_states.shape[-3:])
                histories[t] = shared_histories.reshape((self.real_batch_size, ) + shared_histories.shape[-4:])

                # Start updating all environments with next_actions
                self.runners.update_environments()
                self.runners.wait_updated()
                # Done updating all environments, have new states, rewards and is_over

                episodes_over_masks[t] = 1.0 - shared_episode_over.reshape(
                    (self.real_batch_size, )
                ).astype(np.float32)

                for e_idx, (actual_rewards, episode_overs) in enumerate(zip(shared_rewards, shared_episode_over)):

                    episode_over = episode_overs[0]
                    actual_reward = actual_rewards[0]

                    total_episode_rewards[e_idx] += actual_reward
                    actual_reward = self.rescale_reward(actual_reward)
                    rewards[t, e_idx] = actual_reward

                    emulator_steps[e_idx] += 1
                    self.global_step += 1
                    if episode_over:
                        total_rewards.append(total_episode_rewards[e_idx] / emulator_steps[e_idx])
                        episode_summary = tf.Summary()
                        episode_summary.value.add(simple_value=total_episode_rewards[e_idx], tag="rl/reward")
                        self.summary_writer.add_summary(episode_summary, self.global_step)
                        self.summary_writer.flush()
                        total_episode_rewards[e_idx] = 0
                        emulator_steps[e_idx] = 0

            next_state_value = self.session.run(
                self.network.vs,
                feed_dict={
                    self.network.states: np.reshape(shared_states, (self.real_batch_size, *shared_states.shape[2:])),
                    self.network.history: np.reshape(shared_histories, (self.real_batch_size, *shared_histories.shape[2:])),
                    self.network.agent_positions: np.reshape(shared_positions, (self.real_batch_size, 2))
                }
            )

            estimated_return = np.copy(next_state_value)

            for t in reversed(range(max_local_steps)):
                estimated_return = rewards[t] + self.gamma * estimated_return * episodes_over_masks[t]
                y_batch[t] = np.copy(estimated_return)
                adv_batch[t] = estimated_return - values[t]

            flat_states = states.reshape((self.max_local_steps * self.real_batch_size, *states.shape[2:]))
            flat_history = histories.reshape((self.max_local_steps * self.real_batch_size, *histories.shape[2:]))
            flat_positions = positions.reshape((self.max_local_steps * self.real_batch_size, 2))
            flat_y_batch = y_batch.reshape(-1)
            flat_adv_batch = adv_batch.reshape(-1) / self.network.scale
            flat_actions = actions.reshape(max_local_steps * self.real_batch_size, self.num_actions)
            lr = self.get_lr()
            feed_dict = {
                self.network.states: flat_states,
                self.network.history: flat_history,
                self.network.agent_positions: flat_positions,
                self.network.critic_target: flat_y_batch,
                self.network.actions: flat_actions,
                self.network.advantages: flat_adv_batch,
                self.learning_rate: lr
            }

            _, summaries, global_step = self.session.run(
                [self.train_step, summaries_op, self.network.global_step_tensor],
                feed_dict=feed_dict
            )
            self.summary_writer.add_summary(summaries, global_step)
            self.summary_writer.flush()

            counter += 1

            if counter % (5048 / self.emulator_counts) == 0:
                curr_time = time.time()
                global_steps = self.global_step
                last_ten = 0.0 if len(total_rewards) < 1 else np.mean(total_rewards[-10:])
                logging.info("Ran {} steps, at {} steps/s ({} steps/s avg), last 10 rewards avg {}"
                             .format(global_steps,
                                     self.max_local_steps * self.emulator_counts / (curr_time - loop_start_time),
                                     (global_steps - global_step_start) / (curr_time - start_time),
                                     last_ten))

            self.save_vars()

        coord.join([monitor_thread])
        self.cleanup()

    def _choose_next_actions(self, states, histories, positions):
        states = states.reshape((self.real_batch_size, ) + states.shape[2:])
        histories = histories.reshape((self.real_batch_size, *histories.shape[2:]))
        positions = positions.reshape((self.real_batch_size, 2))
        return self.choose_next_actions(self.network, self.num_actions, states, histories, positions, self.session)

    @staticmethod
    def choose_next_actions(network, num_actions, states, histories, agent_positions, session):
        output = network.predict(states, histories, agent_positions, session)
        network_output_mu, network_output_sigma, network_output_v = output['mu'], output['sigma'], output['vs']
        new_actions = network_output_mu + network_output_sigma * np.random.normal(size=network_output_mu.shape)
        return new_actions, network_output_v
