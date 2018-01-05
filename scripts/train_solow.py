import os
import tensorflow as tf
import itertools
import shutil
import threading
import multiprocessing

import gym

from fed_gym.agents.a3c.estimators import ValueEstimator, GaussianPolicyEstimator, rnn_graph_lstm
from fed_gym.agents.a3c.worker import SolowWorker
from fed_gym.agents.a3c.policy_monitor import PolicyMonitor

tf.logging.set_verbosity(tf.logging.INFO)


tf.flags.DEFINE_string("model_dir", "/tmp/a3c", "Directory to write Tensorboard summaries and videos to.")
tf.flags.DEFINE_integer("t_max", 64, "Number of steps before performing an update")
tf.flags.DEFINE_integer("max_global_steps", 5000, "Stop training after this many steps in the environment. Defaults to running indefinitely.")
tf.flags.DEFINE_integer("eval_every", 5, "Evaluate the policy every N seconds")
tf.flags.DEFINE_boolean("reset", False, "If set, delete the existing model directory and start training from scratch.")
tf.flags.DEFINE_integer("parallelism", None, "Number of threads to run. If not set we run [num_cpu_cores] threads.")

FLAGS = tf.flags.FLAGS
NUM_WORKERS = FLAGS.parallelism if FLAGS.parallelism else multiprocessing.cpu_count()
MODEL_DIR = FLAGS.model_dir
CHECKPOINT_DIR = os.path.join(MODEL_DIR, "checkpoints")

# Optionally empty model directory
if FLAGS.reset:
    shutil.rmtree(MODEL_DIR, ignore_errors=True)

if not os.path.exists(CHECKPOINT_DIR):
    os.makedirs(CHECKPOINT_DIR)

summary_writer = tf.summary.FileWriter(os.path.join(MODEL_DIR, "train"))

INPUT_SIZE = 2
TEMPORAL_SIZE = 1
NUM_ACTIONS = 1


def make_env():
    return gym.envs.make("Solow-v0")

def make_eval_env():
    return gym.envs.make("SolowSS-v0")


with tf.device("/cpu:0"):

    # Keeps track of the number of updates we've performed
    global_step = tf.Variable(0, name="global_step", trainable=False)

    # Global policy and value nets
    with tf.variable_scope("global") as vs:
        policy_net = GaussianPolicyEstimator(
            NUM_ACTIONS, static_size=INPUT_SIZE, temporal_size=TEMPORAL_SIZE,
            shared_layer=lambda x: rnn_graph_lstm(x, 32, 1, True)
        )
        value_net = ValueEstimator(
            static_size=INPUT_SIZE, temporal_size=TEMPORAL_SIZE,
            shared_layer=lambda x: rnn_graph_lstm(x, 32, 1, True),
            reuse=True
        )

    # Global step iterator
    global_counter = itertools.count()

    # Create worker graphs
    workers = []
    for worker_id in xrange(NUM_WORKERS):
        # We only write summaries in one of the workers because they're
        # pretty much identical and writing them on all workers
        # would be a waste of space
        worker_summary_writer = None
        if worker_id == 0:
            worker_summary_writer = summary_writer

        worker = SolowWorker(
            name="worker_{}".format(worker_id),
            env=make_env(),
            policy_net=policy_net,
            value_net=value_net,
            shared_layer=lambda x: rnn_graph_lstm(x, 32, 1, True),
            global_counter=global_counter,
            discount_factor = 0.99,
            summary_writer=worker_summary_writer,
            max_global_steps=FLAGS.max_global_steps
        )
        workers.append(worker)

    saver = tf.train.Saver(keep_checkpoint_every_n_hours=2.0, max_to_keep=10)

    # Used to occasionally save videos for our policy net
    # and write episode rewards to Tensorboard
    pe = PolicyMonitor(
        env=make_eval_env(),
        policy_net=policy_net,
        summary_writer=summary_writer,
        saver=saver,
        num_actions=NUM_ACTIONS,
        input_size=INPUT_SIZE,
        temporal_size=TEMPORAL_SIZE
    )


with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())
    coord = tf.train.Coordinator()

    # Load a previous checkpoint if it exists
    latest_checkpoint = tf.train.latest_checkpoint(CHECKPOINT_DIR)
    if latest_checkpoint:
        print("Loading model checkpoint: {}".format(latest_checkpoint))
        saver.restore(sess, latest_checkpoint)

    # Start worker threads
    worker_threads = []
    for worker in workers:
        t = threading.Thread(target=lambda worker=worker: worker.run(sess, coord, FLAGS.t_max, always_bootstrap=True))
        t.start()
        worker_threads.append(t)

    # Start a thread for policy eval task
    monitor_thread = threading.Thread(target=lambda: pe.continuous_eval(FLAGS.eval_every, sess, coord))
    monitor_thread.start()

    # Wait for all workers to finish
    coord.join(worker_threads)
