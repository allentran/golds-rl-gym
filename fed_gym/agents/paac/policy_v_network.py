from .networks import *
from ..a3c.estimators import make_cell, rnn_graph_lstm


class ConvPolicyVNetwork(ConvNetwork):
    def __init__(self, conf):
        super().__init__(conf)
        self.fc_hidden = 32
        self.rnn_layers = 2

        with tf.device(conf['device']):
            with tf.name_scope(self.name):
                t = tf.shape(self.history)[1]
                n_batches = tf.shape(self.state)[0]
                final_height = int(self.height / (2 ** self.conv_layers))
                final_width = int(self.width / (2 ** self.conv_layers))

                state_idxs = self.action_idxs[:, :-1]

                with tf.variable_scope('process_input'):
                    cnn_history = tf.reshape(self.history, (-1, self.height, self.width, self.channels))
                    cnn_state = self.state
                    for idx in range(self.conv_layers):
                        conv = tf.layers.Conv2D(
                            self.filters,
                            kernel_size=3,
                            padding='same',
                            activation=tf.nn.relu
                        )
                        maxpool = tf.layers.MaxPooling2D(
                            pool_size=(2, 2),
                            strides=2,
                        )

                        cnn_history = maxpool(conv(cnn_history))
                        cnn_state = maxpool(conv(cnn_state))

                    dense1 = tf.layers.Dense(2 * self.fc_hidden, activation=tf.nn.relu)
                    dense2 = tf.layers.Dense(self.fc_hidden, activation=tf.nn.relu)
                    flattened_history = tf.reshape(cnn_history, (n_batches, t, final_height * final_width * self.filters))

                    flattened_state = tf.reshape(cnn_state, (n_batches, final_height * final_width * self.filters))
                    dense_state = dense2(dense1(flattened_state))

                    rnn_cell = tf.nn.rnn_cell.MultiRNNCell([make_cell(self.fc_hidden, True) for _ in range(self.rnn_layers)])
                    outputs, state = tf.nn.dynamic_rnn(
                        rnn_cell, flattened_history, dtype=tf.float32,
                    )
                    rnn_last = state[-1]

                    dense_temporal = tf.layers.dense(rnn_last, 2 * self.fc_hidden, activation=tf.nn.relu)
                    dense_temporal = tf.layers.dense(dense_temporal, self.fc_hidden, activation=tf.nn.relu)

                    self.processed_state = tf.concat([dense_state, dense_temporal], axis=-1)

                with tf.variable_scope('policy'):
                    actions = tf.layers.dense(self.processed_state, 2 * self.fc_hidden, activation=tf.nn.relu)
                    actions = tf.layers.dense(
                        actions, self.height * self.width * self.num_actions, activation=tf.nn.relu
                    )
                    actions = tf.layers.dense(
                        actions, self.height * self.width * self.num_actions, activation=tf.nn.relu
                    )
                    actions = tf.reshape(actions, (n_batches, self.height, self.width, self.num_actions))
                    action_probs = tf.nn.softmax(actions)

                self.probs = tf.gather_nd(action_probs, state_idxs)
                selected_probs = tf.gather_nd(action_probs, self.action_idxs)
                log_probs = tf.log(selected_probs + tf.keras.backend.epsilon())

                self.entropy = - tf.reduce_sum(
                    self.probs * tf.log(self.probs+ tf.keras.backend.epsilon()),
                    axis=-1
                )

                self.policy_loss = - tf.reduce_mean(log_probs * self.advantages + self.entropy_beta * self.entropy)

                with tf.variable_scope('v_s'):
                    vs = tf.layers.dense(self.processed_state, self.fc_hidden * 2, activation=tf.nn.relu)
                    vs = tf.layers.dense(vs, self.fc_hidden, activation=tf.nn.relu)
                    vs = tf.layers.dense(vs, self.height * self.width, activation=tf.nn.relu)
                    vs = tf.reshape(vs, (n_batches, self.height, self.width))

                self.vs = tf.gather_nd(vs, state_idxs)
                self.critic_loss = tf.squared_difference(self.vs, self.critic_target)
                self.critic_loss_mean = tf.reduce_mean(0.25 * self.critic_loss, name='mean_critic_loss')

                # Loss scaling is used because the learning rate was initially runed tuned to be used with
                # max_local_steps = 5 and summing over timesteps, which is now replaced with the mean.
                self.loss = self.policy_loss + self.critic_loss_mean


class FlatPolicyVNetwork(FlatNetwork):

    def __init__(self, conf):
        """ Set up remaining layers, objective and loss functions, gradient
        compute and apply ops, network parameter synchronization ops, and
        summary ops. """

        super(FlatPolicyVNetwork, self).__init__(conf)

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

