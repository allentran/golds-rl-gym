import time
from multiprocessing.sharedctypes import RawArray
from ctypes import c_float

from fed_gym.agents.a3c.estimators import SolowStateProcessor
from .actor_learner import *
import logging

from ..a3c.worker import sigmoid
from .runners import Runners
import numpy as np


class PAACLearner(ActorLearner):
    def __init__(self, network_creator, environment_creator, args):
        super(PAACLearner, self).__init__(network_creator, environment_creator, args)
        self.workers = args.emulator_workers
        self.rnn_length = args.rnn_length
        self.state_processor = SolowStateProcessor()

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

    def __choose_next_actions(self, states, histories):
        return PAACLearner.choose_next_actions(self.network, self.num_actions, states, histories, self.session)

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

        self.runners = Runners(self.emulators, self.workers, variables)
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
                next_actions, readouts_v_t = self.__choose_next_actions(shared_states, shared_histories)
                for z in range(next_actions.shape[0]):
                    shared_actions[z] = sigmoid(next_actions[z])

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
                        episode_summary = tf.Summary(value=[
                            tf.Summary.Value(tag='rl/reward', simple_value=total_episode_rewards[e_idx]),
                            tf.Summary.Value(tag='rl/episode_length', simple_value=emulator_steps[e_idx]),
                        ])
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

            _, summaries = self.session.run(
                [self.train_step, summaries_op],
                feed_dict=feed_dict)

            self.summary_writer.add_summary(summaries, self.global_step)
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
