""" run Neural RL methods on partially observable environments """
from argparse import ArgumentParser
from episode import run_episode

from environments import tiger
from environments import cartpole

from agents.dqn import DQN

import numpy as np # [fixme] remove at some point

def main():
    """ start of program """

    conf = parse_arguments()
    env = get_environment(conf)
    agent = DQN(env, conf)

    run = 1
    returns = []
    while True:
        print("running episode " + str(run))

        returns.append(run_episode(env, agent, conf))

        run = run + 1
        print('mean return: ' + str(np.mean(returns[-100:])))

def parse_arguments():
    """ parse command line arguments, returns a namespace with all variables"""
    parser = ArgumentParser()

    parser.add_argument(
        "--method",
        help="which learning method to use (dqn)",
        required=True)

    parser.add_argument(
        "--domain", "-D",
        help="which domain to use method on",
        required=True,
        choices=["cartpole", "tiger"])

    parser.add_argument(
        "--discount", "-d",
        default=1,
        type=float,
        help="the discount factor used in the domain")

    parser.add_argument(
        "--observation_len",
        default=1,
        type=int,
        help="number of past observations to provide to the policy")

    parser.add_argument(
        "--replay_buffer_size", "--rb_size",
        default=1000000,
        type=int,
        help="size of replay buffer")

    parser.add_argument(
        "--batch_size",
        default=32,
        type=int,
        help="size of learning batch")

    parser.add_argument(
        "--gamma",
        default=0.99,
        type=float,
        help="discount factor to be used")

    parser.add_argument(
        "--q_target_update_freq",
        default=250,
        type=int,
        help="how often the target network should be updated (every # time steps)")

    parser.add_argument(
        "--train_frequency",
        default=4,
        type=int,
        help="how often the agent performs a batch update (every # time steps)")

    parser.add_argument(
        "--explore_duration",
        default=10000,
        type=int,
        help="how long to wait before training (# time steps)")

    parser.add_argument(
        "--learning_rate", "--alpha",
        default=1e-4,
        type=float,
        help="learning rate of the policy gradient descent")

    return parser.parse_args()

def get_environment(conf):
    """ returns environments as indicated in conf """
    if conf.domain == "tiger":
        return tiger.Tiger()
    if conf.domain == "cartpole":
        return cartpole.Cartpole()

if __name__ == '__main__':
    main()
