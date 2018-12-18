""" run Neural RL methods on partially observable environments """
from argparse import ArgumentParser
from argparse import ArgumentDefaultsHelpFormatter
import time
from math import sqrt

from episode import run_episode

from environments import tiger
from environments import cartpole

from agents.dqn import DQN

import numpy as np

def main():
    """ start of program """

    conf = parse_arguments()

    cur_time = time.time()
    result = np.zeros((conf.runs, conf.episodes))
    for run in range(conf.runs):

        if conf.verbose:
            print('starting run', run)

        env = get_environment(conf)
        agent = DQN(env, conf)

        for episode in range(conf.episodes):

            result[run][episode] = run_episode(env, agent, conf)

            if  conf.verbose and time.time() - cur_time > 10:

                print(time.ctime(),
                      "run", run, "episode", episode,
                      ": avg return",
                      np.mean(result[run, max(0, episode-100):episode]))

                cur_time = time.time()

    # process results into rows of (for each episode
    # return avg, return var, return #, return stder
    summary = np.transpose([
        np.mean(result, axis=0),
        np.var(result, axis=0),
        [conf.runs] * conf.episodes, \
        np.var(result, axis=0) / sqrt(conf.runs)
        ])

    np.savetxt(
        conf.file,
        summary,
        delimiter=', ',
        header="version 1:\nreturn mean, return var, return count, return stder")


def parse_arguments():
    """ parse command line arguments, returns a namespace with all variables"""
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        "--verbose", "-v",
        action='store_true',
        help="whether to output verbose messages")

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
        "--file", "-f",
        default="output.npy",
        help="output file path")

    parser.add_argument(
        "--runs",
        default=1,
        type=int,
        help="number of runs to average returns over")

    parser.add_argument(
        "--episodes",
        default=100000,
        type=int,
        help="number of episodes to run")

    parser.add_argument(
        "--network_size",
        required=True,
        help='the size of the q-network',
        choices=["small", "med", "large"])

    parser.add_argument(
        "--learning_rate", "--alpha",
        default=1e-4,
        type=float,
        help="learning rate of the policy gradient descent")

    parser.add_argument(
        "--discount", "-d",
        default=1,
        type=float,
        help="the discount factor used in the domain")

    parser.add_argument(
        "--gamma",
        default=0.99,
        type=float,
        help="discount factor to be used")

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

    return parser.parse_args()

def get_environment(conf):
    """ returns environments as indicated in conf """
    if conf.domain == "tiger":
        return tiger.Tiger(conf)
    if conf.domain == "cartpole":
        return cartpole.Cartpole(conf)

if __name__ == '__main__':
    main()
