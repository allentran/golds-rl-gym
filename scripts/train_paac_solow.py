import argparse
import logging
import sys
import signal
import os
import copy

import tensorflow as tf

from fed_gym.agents.paac import environment_creator
from fed_gym.agents.paac.paac import PAACLearner
from fed_gym.agents.paac.policy_v_network import FlatPolicyVNetwork
from fed_gym.envs.fed_env import register_solow_env
from fed_gym.agents.state_processors import SolowStateProcessor
from fed_gym.agents.paac.emulator_runner import SolowRunner

register_solow_env(1, 1)
logging.basicConfig(stream=sys.stdout, level=logging.INFO)


def bool_arg(string):
    value = string.lower()
    if value == 'true':
        return True
    elif value == 'false':
        return False
    else:
        raise argparse.ArgumentTypeError("Expected True or False, but got {}".format(string))


def main(args):
    logging.debug('Configuration: {}'.format(args))

    network_creator, env_creator = get_network_and_environment_creator(args)

    learner = PAACLearner(network_creator, env_creator, args, SolowRunner, SolowStateProcessor())

    setup_kill_signal_handler(learner)

    logging.info('Starting training')
    learner.train()
    logging.info('Finished training')


def setup_kill_signal_handler(learner):
    main_process_pid = os.getpid()

    def signal_handler(signal, frame):
        if os.getpid() == main_process_pid:
            logging.info('Signal ' + str(signal) + ' detected, cleaning up.')
            learner.cleanup()
            logging.info('Cleanup completed, shutting down...')
            sys.exit(0)

    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)


def get_network_and_environment_creator(args, random_seed=3):
    env_creator = environment_creator.SolowEnvironmentCreator(1, 1)
    num_actions = env_creator.num_actions
    args.num_actions = num_actions
    args.random_seed = random_seed

    network_conf = {
        'num_actions': num_actions,
        'entropy_regularisation_strength': args.entropy_regularisation_strength,
        'device': args.device,
        'scale': args.scale,
        'clip_norm': args.clip_norm,
        'clip_norm_type': args.clip_norm_type,
        'static_size': args.static_size,
        'temporal_size': args.temporal_size,
        'static_hidden_size': args.static_hidden_size,
        'rnn_hidden_size': args.temporal_hidden_size,
    }

    def network_creator(name='local_learning'):
        nonlocal network_conf
        copied_network_conf = copy.copy(network_conf)
        copied_network_conf['name'] = name
        with tf.variable_scope("global"):
            return FlatPolicyVNetwork(copied_network_conf)

    return network_creator, env_creator


def get_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--device', default='/gpu:0', type=str, help="Device to be used ('/cpu:0', '/gpu:0', '/gpu:1',...)", dest="device")
    parser.add_argument('--e', default=0.1, type=float, help="Epsilon for the Rmsprop and Adam optimizers", dest="e")
    parser.add_argument('--alpha', default=0.99, type=float, help="Discount factor for the history/coming gradient, for the Rmsprop optimizer", dest="alpha")
    parser.add_argument('-lr', '--initial_lr', default=0.0001, type=float, help="Initial value for the learning rate. Default = 1e-4", dest="initial_lr")
    parser.add_argument('-lra', '--lr_annealing_steps', default=80000000, type=int, help="Nr. of global steps during which the learning rate will be linearly annealed towards zero", dest="lr_annealing_steps")
    parser.add_argument('--entropy', default=0.02, type=float, help="Strength of the entropy regularization term (needed for actor-critic)", dest="entropy_regularisation_strength")
    parser.add_argument('--clip_norm', default=40.0, type=float, help="If clip_norm_type is local/global, grads will be clipped at the specified maximum (avaerage) L2-norm", dest="clip_norm")
    parser.add_argument('--clip_norm_type', default="global", help="Whether to clip grads by their norm or not. Values: ignore (no clipping), local (layer-wise norm), global (global norm)", dest="clip_norm_type")
    parser.add_argument('--gamma', default=0.99, type=float, help="Discount factor", dest="gamma")
    parser.add_argument('--max_global_steps', default=80000000, type=int, help="Max. number of training steps", dest="max_global_steps")
    parser.add_argument('--max_local_steps', default=5, type=int, help="Number of steps to gain experience from before every update.", dest="max_local_steps")
    parser.add_argument('--single_life_episodes', default=False, type=bool_arg, help="If True, training episodes will be terminated when a life is lost (for games)", dest="single_life_episodes")
    parser.add_argument('-ec', '--emulator_counts', default=32, type=int, help="The amount of emulators per agent. Default is 32.", dest="emulator_counts")
    parser.add_argument('-ew', '--emulator_workers', default=8, type=int, help="The amount of emulator workers per agent. Default is 8.", dest="emulator_workers")
    parser.add_argument('-df', '--debugging_folder', default='logs/', type=str, help="Folder where to save the debugging information.", dest="debugging_folder")
    parser.add_argument('-rs', '--random_start', default=True, type=bool_arg, help="Whether or not to start with 30 noops for each env. Default True", dest="random_start")
    parser.add_argument('--scale', default=100., type=float)
    parser.add_argument('--rnn-length', default=5, type=int)
    parser.add_argument('--static-size', default=2, type=int)
    parser.add_argument('--temporal-size', default=2, type=int)
    parser.add_argument('--static-hidden-size', default=32, type=int)
    parser.add_argument('--temporal-hidden-size', default=32, type=int)
    return parser


if __name__ == '__main__':
    args = get_arg_parser().parse_args()

    main(args)
