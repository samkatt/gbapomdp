""" ensemble of q-nets """

import numpy as np
import tensorflow as tf

import agents.agent as agents
import agents.networks.replay_buffer as rb


class EnsembleAgent(agents.Agent):
    """ ensemble agent """

    t = 0
    last_ob = 0
    _storing_rbs = 0

    # the policy from which to act e-greedy on right now
    _current_policy = 0

    # FIXME: take specific arguments instead of conf
    def __init__(self,
                 qnet_constructor,
                 arch,
                 env,
                 conf,
                 name='ensemble-agent'):
        """ initialize network """

        if conf.num_nets == 1:
            raise ValueError("no number of networks specified (--num_nets)")

        # consts
        optimizer = tf.train.AdamOptimizer(learning_rate=conf.learning_rate)

        # confs
        self.target_update_freq = conf.q_target_update_freq
        self.batch_size = conf.batch_size
        self.train_freq = conf.train_frequency

        # construct the replay buffer
        self.replay_buffers = np.array([
            {'index': 0, 'buffer': rb.ReplayBuffer(
                conf.replay_buffer_size,
                conf.observation_len, True)}
            for _ in range(conf.num_nets)
        ])

        # which buffers are collecting the current episode
        _storing_rbs = np.zeros(conf.num_nets).astype(bool)

        self.nets = [
            qnet_constructor(
                env.spaces(),
                arch,
                optimizer,
                conf,
                name + '_net_' + str(i))
            for i in range(conf.num_nets)
        ]

    def reset(self, obs):
        """ resets to finish episode """

        self.last_ob = obs

        for net in self.nets:
            net.reset()

        self._current_policy = np.random.randint(0, len(self.nets) - 1)

        # update which buffers are storing this episode
        self._storing_rbs = np.random.rand(len(self.replay_buffers)) > .5
        # make sure current is tracking
        self._storing_rbs[self._current_policy] = True

        for rb in self.replay_buffers[self._storing_rbs]:
            rb['index'] = rb['buffer'].store_frame(self.last_ob)

    def select_action(self):
        """ requests greedy action from network """

        q_in = np.array([self.last_ob]) if self.nets[self._current_policy].is_recurrent(
        ) else self.replay_buffers[self._current_policy]['buffer'].encode_recent_observation()

        q_values = self.nets[self._current_policy].qvalues(q_in)

        self.latest_action = q_values.argmax()

        return self.latest_action

    def update(self, obs, reward, terminal):
        """update informs agent of observed transition

        For each network:

            Stores the experience (together with stored action) into the buffer
            with some probability

            May perform a batch update (every so often, see
            parameters/configuration)

            May update the target network (every so often, see
            parameters/configuration)


        :param _observation: the observation from the last step
        :param _reward: the reward of the last step
        :param _terminal: whether the last step was terminal
        """

        # store experience
        for rb in self.replay_buffers[self._storing_rbs]:
            rb['buffer'].store_effect(
                rb['index'],
                self.latest_action,
                reward,
                terminal
            )

        self.last_ob = obs

        # store experience
        for rb in self.replay_buffers[self._storing_rbs]:
            rb['index'] = rb['buffer'].store_frame(self.last_ob)

        # batch update all networks using there respective replay buffer
        for net, rb in zip(self.nets, self.replay_buffers):
            if rb['buffer'].can_sample(
                    self.batch_size) and self.t % self.train_freq == 0:

                obs, actions, rewards, next_obs, done_mask = \
                    rb['buffer'].sample(self.batch_size)

                net.batch_update(obs, actions, rewards, next_obs, done_mask)

        # update all target networks occasionally
        if self.t % self.target_update_freq == 0:
            for net in self.nets:
                net.update_target()

        self.t = self.t + 1
