import tensorflow as tf
import logging
import numpy as np


def flatten(_input):
    shape = _input.get_shape().as_list()
    dim = shape[1]*shape[2]*shape[3]
    return tf.reshape(_input, [-1,dim], name='_flattened')


def conv2d(name, _input, filters, size, channels, stride, padding = 'VALID', init = "torch"):
    w = conv_weight_variable([size,size,channels,filters],
                             name + '_weights', init = init)
    b = conv_bias_variable([filters], size, size, channels,
                           name + '_biases', init = init)
    conv = tf.nn.conv2d(_input, w, strides=[1, stride, stride, 1],
                        padding=padding, name=name + '_convs')
    out = tf.nn.relu(tf.add(conv, b),
                     name='' + name + '_activations')
    return w, b, out


def conv_weight_variable(shape, name, init = "torch"):
    if init == "glorot_uniform":
        receptive_field_size = np.prod(shape[:2])
        fan_in = shape[-2] * receptive_field_size
        fan_out = shape[-1] * receptive_field_size
        d = np.sqrt(6. / (fan_in + fan_out))
    else:
        w = shape[0]
        h = shape[1]
        input_channels = shape[2]
        d = 1.0 / np.sqrt(input_channels * w * h)

    initial = tf.random_uniform(shape, minval=-d, maxval=d)
    return tf.Variable(initial, name=name, dtype='float32')


def conv_bias_variable(shape, w, h, input_channels, name, init= "torch"):
    if init == "glorot_uniform":
        initial = tf.zeros(shape)
    else:
        d = 1.0 / np.sqrt(input_channels * w * h)
        initial = tf.random_uniform(shape, minval=-d, maxval=d)
    return tf.Variable(initial, name=name, dtype='float32')


def fc(name, _input, output_dim, activation = "relu", init = "torch"):
    input_dim = _input.get_shape().as_list()[1]
    w = fc_weight_variable([input_dim, output_dim],
                           name + '_weights', init = init)
    b = fc_bias_variable([output_dim], input_dim,
                         '' + name + '_biases', init = init)
    out = tf.add(tf.matmul(_input, w), b, name= name + '_out')

    if activation == "relu":
        out = tf.nn.relu(out, name='' + name + '_relu')

    return w, b, out


def fc_weight_variable(shape, name, init="torch"):
    if init == "glorot_uniform":
        fan_in = shape[0]
        fan_out = shape[1]
        d = np.sqrt(6. / (fan_in + fan_out))
    else:
        input_channels = shape[0]
        d = 1.0 / np.sqrt(input_channels)
    initial = tf.random_uniform(shape, minval=-d, maxval=d)
    return tf.Variable(initial, name=name, dtype='float32')


def fc_bias_variable(shape, input_channels, name, init= "torch"):
    if init=="glorot_uniform":
        initial = tf.zeros(shape, dtype='float32')
    else:
        d = 1.0 / np.sqrt(input_channels)
        initial = tf.random_uniform(shape, minval=-d, maxval=d)
    return tf.Variable(initial, name=name, dtype='float32')


def softmax(name, _input, output_dim):
    input_dim = _input.get_shape().as_list()[1]
    w = fc_weight_variable([input_dim, output_dim], name + '_weights')
    b = fc_bias_variable([output_dim], input_dim, name + '_biases')
    out = tf.nn.softmax(tf.add(tf.matmul(_input, w), b), name= name + '_policy')
    return w, b, out


def log_softmax( name, _input, output_dim):
    input_dim = _input.get_shape().as_list()[1]
    w = fc_weight_variable([input_dim, output_dim], name + '_weights')
    b = fc_bias_variable([output_dim], input_dim, name + '_biases')
    out = tf.nn.log_softmax(tf.add(tf.matmul(_input, w), b), name= name + '_policy')
    return w, b, out


class Network(object):

    def __init__(self, conf):

        self.name = conf['name']
        self.num_actions = conf['num_actions']
        self.clip_norm = conf['clip_norm']
        self.clip_norm_type = conf['clip_norm_type']
        self.device = conf['device']
        self.static_size = conf['static_size']
        self.entropy_beta = conf['entropy_regularisation_strength']
        self.scale = conf['scale']

        with tf.device(self.device):
            with tf.name_scope(self.name):
                self.critic_target = tf.placeholder("float32", (None, ), name='target')
                self.advantages = tf.placeholder("float", (None, ), name='advantage')

        # The output layer
        self.output = None

    def init(self, checkpoint_folder, saver, session):
        last_saving_step = 0

        with tf.device('/cpu:0'):
            # Initialize network parameters
            path = tf.train.latest_checkpoint(checkpoint_folder)
            path = None
            if path is None:
                logging.info('Initializing all variables')
                session.run(tf.global_variables_initializer())
            else:
                logging.info('Restoring network variables from previous run')
                saver.restore(session, path)
                last_saving_step = int(path[path.rindex('-')+1:])
        return last_saving_step


class FlatNetwork(Network):
    def __init__(self, conf):
        super().__init__(conf)
        self.temporal_size = conf['temporal_size']

        with tf.device(self.device):
            with tf.name_scope(self.name):
                self.actions = tf.placeholder(shape=(None, self.num_actions), dtype=tf.float32, name="actions")
                self.states = tf.placeholder(shape=(None, self.static_size), dtype=tf.float32, name="X")
                self.history = tf.placeholder(shape=(None, None, self.temporal_size), dtype=tf.float32, name="X_t")


class ConvNetwork(Network):
    def __init__(self, conf):
        super().__init__(conf)

        self.height = conf['height']
        self.width = conf['width']
        self.channels = conf['channels']
        self.filters = conf['filters']
        self.conv_layers = conf['conv_layers']

        with tf.device(self.device):
            with tf.name_scope(self.name):
                # action idxs are gather_nd tuples [(batch_idx, height_idx, width_idx, action_idx)]
                self.action_idxs = tf.placeholder(shape=(None, 4), dtype=tf.int32, name="actions")
                self.state = tf.placeholder(
                    shape=(None, self.height, self.width, self.channels), dtype=tf.float32, name="X"
                )
                self.history = tf.placeholder(
                    shape=(None, None, self.height, self.width, self.channels), dtype=tf.float32, name="X_t"
                )
