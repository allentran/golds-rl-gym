from .networks import *
from ..a3c.estimators import rnn_graph_lstm


class PolicyVNetwork(Network):

    def __init__(self, conf):
        """ Set up remaining layers, objective and loss functions, gradient
        compute and apply ops, network parameter synchronization ops, and
        summary ops. """

        super(PolicyVNetwork, self).__init__(conf)

        self.critic_target = tf.placeholder("float32", (None, ), name='target')
        self.advantages = tf.placeholder("float", (None, ), name='advantage')

        self.entropy_regularisation_strength = conf['entropy_regularisation_strength']
        static_hidden_size = conf['static_hidden_size']
        rnn_hidden_size = conf['rnn_hidden_size']
        num_actions = conf['num_actions']
        ub = 5.
        lb = -5.

        with tf.device(conf['device']):
            with tf.name_scope(self.name):

                with tf.variable_scope('process_input'):
                    dense_output = rnn_graph_lstm(self.history, self.states, rnn_hidden_size, 1, True)

                with tf.variable_scope('mu'):
                    mu = tf.layers.dense(dense_output, static_hidden_size * 2, activation=tf.nn.relu)
                    mu = tf.layers.dense(mu, static_hidden_size, activation=tf.nn.tanh)
                    self.mu = ((ub - lb) / 2.) * tf.layers.dense(mu, num_actions, activation=tf.nn.tanh) + ((lb + ub) / 2.)

                with tf.variable_scope('sigma'):
                    sigma = tf.layers.dense(dense_output, static_hidden_size * 2, activation=tf.nn.relu)
                    sigma = tf.layers.dense(sigma, static_hidden_size, activation=tf.nn.tanh)
                    self.sigma = tf.layers.dense(
                        sigma, num_actions, activation=tf.nn.sigmoid, bias_initializer=tf.constant_initializer(-1.)
                    ) + 1e-3

                dist = tf.distributions.Normal(loc=self.mu, scale=self.sigma)

                self.entropy = dist.entropy()
                self.entropy_mean = tf.reduce_mean(self.entropy, name="entropy_mean")

                nll = - dist.log_prob(self.actions)
                loss = nll * self.advantages[:, None]
                self.policy_loss = tf.identity(tf.reduce_mean(loss), name='loss')

                with tf.variable_scope('v_s'):
                    vs = tf.layers.dense(dense_output, static_hidden_size * 2, activation=tf.nn.tanh)
                    self.vs = self.scale * tf.layers.dense(
                        inputs=vs,
                        units=1,
                        activation=None,
                    )
                    self.vs = tf.squeeze(self.vs, squeeze_dims=[1], name="logits")

                self.critic_loss = tf.squared_difference(self.vs, self.critic_target)
                self.critic_loss_mean = tf.reduce_mean(0.25 * self.critic_loss, name='mean_critic_loss')

                # Loss scaling is used because the learning rate was initially runed tuned to be used with
                # max_local_steps = 5 and summing over timesteps, which is now replaced with the mean.
                self.loss = self.policy_loss + self.critic_loss_mean

