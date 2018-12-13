""" run Neural RL methods on partially observable environments """
from argparse import ArgumentParser
from episode import run_episode
from environments.tiger import Tiger
from agents.dqn import DQN

def main():
    """ start of program """

    conf = parse_arguments()

    run_episode(Tiger(), DQN(), conf)

    # session = utils.get_session()

    # env = gym.make('CartPole-v0')
    # env = gym.wrappers.Monitor(env, 'videos/', force=True)

    # seed = 0
    # utils.set_global_seeds(seed)
    # env.seed(seed)

    # optimizer = tf.train.AdamOptimizer(learning_rate=1e-4)

    # n_timesteps = 1000000
    # exploration_schedule = utils.PiecewiseSchedule(
                               # [(0, 1.0), (2e5, 0.1)],
                               # outside_value=0.1,
                           # )

    # dqn.learn(
        # env,
        # q_func=CartPoleNet(),
        # optimizer=optimizer,
        # session=session,
        # exploration=exploration_schedule,
        # max_timesteps=n_timesteps,
        # replay_buffer_size=1000000,
        # batch_size=32,
        # gamma=0.99,
        # learning_starts=10000,
        # learning_freq=4,
        # history_len=1,
        # target_update_freq=250,
        # use_float=True,
        # log_every_n_steps=25000,
    # )
    # env.close()


def parse_arguments():
    """ parse command line arguments, returns a namespace with all variables"""
    parser = ArgumentParser()

    parser.add_argument(
            "--method",
            dest="method",
            help="which learning method to use (dqn)",
            required=True
            )

    parser.add_argument(
            "-d", "--discount",
            dest="discount",
            default = 0.99,
            help="the discount factor used in the domain"
            )

    return parser.parse_args()


if __name__ == '__main__':
    main()
