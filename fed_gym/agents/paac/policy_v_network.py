from .networks import *
from ..a3c.estimators import make_cell, rnn_graph_lstm


class ConvSingleAgentPolicyNetwork(ConvSingleAgentNetwork):

    def __init__(self, conf):
        super().__init__(conf)
        self.fc_hidden = 256

        with tf.device(conf['device']):
            with tf.name_scope(self.name):
                with tf.variable_scope('process_input'):
                    conv = tf.layers.Conv2D(
                        32,
                        kernel_size=8,
                        strides=4,
                        activation=tf.nn.relu
                    )(self.states)
                    conv = tf.layers.Conv2D(
                        64,
                        kernel_size=4,
                        strides=2,
                        activation=tf.nn.relu
                    )(conv)
                    conv = tf.layers.Conv2D(
                        64,
                        kernel_size=3,
                        activation=tf.nn.relu
                    )(conv)

                    dense1 = tf.layers.Dense(2 * self.fc_hidden, activation=tf.nn.relu)
                    dense2 = tf.layers.Dense(self.fc_hidden, activation=tf.nn.relu)

                    flattened_state = tf.layers.flatten(conv)
                    self.processed_state = dense2(dense1(flattened_state))

                with tf.variable_scope('policy'):
                    actions = tf.layers.dense(self.processed_state, 2 * self.fc_hidden, activation=tf.nn.relu)
                    actions = tf.layers.dense(
                        actions, self.height * self.width * self.num_actions * 2, activation=tf.nn.relu
                    )
                    self.mu = tf.layers.dense(
                        actions, self.num_actions, activation=tf.nn.tanh
                    )
                    self.sigma = tf.layers.dense(
                        actions, self.num_actions, activation=tf.nn.sigmoid
                    )

                normal_dist = tf.distributions.Normal(self.mu, self.sigma)

                log_l = normal_dist.log_prob(self.actions)
                self.entropy = normal_dist.entropy()
                if self.num_actions > 1:
                    log_l = tf.reduce_sum(log_l, axis=-1)
                    self.entropy = tf.reduce_sum(self.entropy, axis=-1)
                self.policy_loss = - tf.reduce_mean(log_l * self.advantages + self.entropy_beta * self.entropy)

                with tf.variable_scope('v_s'):
                    vs = tf.layers.dense(self.processed_state, self.fc_hidden * 2, activation=tf.nn.relu)
                    vs = tf.layers.dense(vs, self.fc_hidden, activation=tf.nn.relu)
                    self.vs = - self.scale * tf.squeeze(tf.layers.dense(vs, 1, activation=tf.nn.softplus))

                self.critic_loss = tf.squared_difference(self.vs, self.critic_target) / self.scale
                self.critic_loss_mean = tf.reduce_mean(0.25 * self.critic_loss, name='mean_critic_loss')

                # Loss scaling is used because the learning rate was initially runed tuned to be used with
                # max_local_steps = 5 and summing over timesteps, which is now replaced with the mean.
                self.loss = self.policy_loss + self.critic_loss_mean


    def predict(self, states, session):
        feed_dict = {
            self.states: states,
        }
        return session.run(
            {
                'vs': self.vs,
                'mu': self.mu,
                'sigma': self.sigma,
            },
            feed_dict
        )


class ConvPolicyVFieldNetwork(ConvFieldNetwork):
    def __init__(self, conf):
        super().__init__(conf)
        self.fc_hidden = 32
        self.rnn_layers = 2
        self.use_rnn = False

        with tf.device(conf['device']):
            with tf.name_scope(self.name):
                t = tf.shape(self.history)[1]
                n_batches = tf.shape(self.states)[0]
                final_height = int(self.height / (2 ** self.conv_layers))
                final_width = int(self.width / (2 ** self.conv_layers))

                with tf.variable_scope('process_input'):
                    if self.use_rnn:
                        cnn_history = tf.reshape(self.history, (-1, self.height, self.width, self.channels))
                    cnn_state = self.states
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

                        if self.use_rnn:
                            cnn_history = maxpool(conv(cnn_history))
                        cnn_state = maxpool(conv(cnn_state))

                    dense1 = tf.layers.Dense(2 * self.fc_hidden, activation=tf.nn.relu)
                    dense2 = tf.layers.Dense(self.fc_hidden, activation=tf.nn.relu)

                    flattened_state = tf.reshape(cnn_state, (n_batches, final_height * final_width * self.filters))
                    dense_state = dense2(dense1(flattened_state))

                    if self.use_rnn:
                        flattened_history = tf.reshape(cnn_history, (n_batches, t, final_height * final_width * self.filters))
                        rnn_cell = tf.nn.rnn_cell.MultiRNNCell([make_cell(self.fc_hidden, True) for _ in range(self.rnn_layers)])
                        outputs, state = tf.nn.dynamic_rnn(
                            rnn_cell, flattened_history, dtype=tf.float32,
                        )
                        rnn_last = state[-1]

                        dense_temporal = tf.layers.dense(rnn_last, 2 * self.fc_hidden, activation=tf.nn.relu)
                        dense_temporal = tf.layers.dense(dense_temporal, self.fc_hidden, activation=tf.nn.relu)

                        self.processed_state = tf.concat([dense_state, dense_temporal], axis=-1)
                    else:
                        self.processed_state = dense_state

                with tf.variable_scope('policy'):
                    actions = tf.layers.dense(self.processed_state, 2 * self.fc_hidden, activation=tf.nn.relu)
                    actions = tf.layers.dense(
                        actions, self.height * self.width * self.num_actions * 2, activation=tf.nn.relu
                    )
                    mus = tf.layers.dense(
                        actions, self.height * self.width * self.num_actions, activation=tf.nn.tanh
                    )
                    mus = tf.reshape(mus, (n_batches, self.height, self.width, self.num_actions))
                    sigmas = tf.layers.dense(
                        actions, self.height * self.width * self.num_actions, activation=tf.nn.sigmoid
                    )
                    sigmas = tf.reshape(sigmas, (n_batches, self.height, self.width, self.num_actions))

                agent_positions = tf.concat([tf.range(n_batches)[:, None], self.agent_positions], axis=-1)

                self.mu = tf.gather_nd(mus, agent_positions)
                self.sigma = tf.gather_nd(sigmas, agent_positions)

                normal_dist = tf.distributions.Normal(self.mu, self.sigma)

                log_l = normal_dist.log_prob(self.actions)
                self.entropy = normal_dist.entropy()
                if self.num_actions > 1:
                    log_l = tf.reduce_sum(log_l, axis=-1)
                    self.entropy = tf.reduce_sum(self.entropy, axis=-1)
                self.policy_loss = - tf.reduce_mean(log_l * self.advantages + self.entropy_beta * self.entropy)

                with tf.variable_scope('v_s'):
                    vs = tf.layers.dense(self.processed_state, self.fc_hidden * 2, activation=tf.nn.relu)
                    vs = tf.layers.dense(vs, self.fc_hidden, activation=tf.nn.relu)
                    self.vs = - self.scale * tf.squeeze(tf.layers.dense(vs, 1, activation=tf.nn.softplus))

                self.critic_loss = tf.squared_difference(self.vs, self.critic_target) / self.scale
                self.critic_loss_mean = tf.reduce_mean(0.25 * self.critic_loss, name='mean_critic_loss')

                # Loss scaling is used because the learning rate was initially runed tuned to be used with
                # max_local_steps = 5 and summing over timesteps, which is now replaced with the mean.
                self.loss = self.policy_loss + self.critic_loss_mean

    def predict(self, states, histories, positions, session):
        feed_dict = {
            self.states: states,
            # self.history: histories,
            self.agent_positions: positions
        }
        return session.run(
            {
                'vs': self.vs,
                'mu': self.mu,
                'sigma': self.sigma,
            },
            feed_dict
        )


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

                self.critic_loss = tf.squared_difference(self.vs, self.critic_target) / self.scale
                self.critic_loss_mean = tf.reduce_mean(0.25 * self.critic_loss, name='mean_critic_loss')

                # Loss scaling is used because the learning rate was initially runed tuned to be used with
                # max_local_steps = 5 and summing over timesteps, which is now replaced with the mean.
                self.loss = self.policy_loss + self.critic_loss_mean

    def predict(self, states, histories, session):
        feed_dict = {
            self.states: states,
            self.history: histories,
        }
        return session.run(
            {
                'mu': self.mu,
                'sigma': self.sigma,
            },
            feed_dict
        )
