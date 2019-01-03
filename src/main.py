""" run Neural RL methods on partially observable environments """
from argparse import ArgumentParser
from argparse import ArgumentDefaultsHelpFormatter
import time
from math import sqrt
import numpy as np

from utils import tf_wrapper

from episode import run_episode

from environments import tiger
from environments import cartpole
from environments import gridworld

from agents.dqn import DQN
from agents.ensemble_dqn import ensemble_DQN


def main():
    """ start of program """

    conf = parse_arguments()

    env = get_environment(conf)

    cur_time = time.time()
    result_mean = np.zeros(conf.episodes)
    result_var = np.zeros(conf.episodes)

    with tf_wrapper.get_session() as sess:
        for run in range(conf.runs):

            print('starting run', run)

            agent = get_agent(conf, env, sess, 'agent-' + str(run))
            tmp_res = np.zeros(conf.episodes)

            for episode in range(conf.episodes):

                tmp_res[episode] = run_episode(env, agent, conf)

                if  episode > 0 and conf.verbose and time.time() - cur_time > 5:

                    print(time.ctime(),
                          "run", run, "episode", episode,
                          ": avg return",
                          np.mean(tmp_res[max(0, episode-100):episode]))

                    cur_time = time.time()

            # update mean and variance
            delta = tmp_res - result_mean
            result_mean += delta / (run+1)
            delta_2 = tmp_res - result_mean
            result_var += delta * delta_2

            # process results into rows of for each episode
            # return avg, return var, return #, return stder
            summary = np.transpose([
                result_mean,
                result_var,
                [run+1] * conf.episodes,
                result_var / sqrt(conf.runs)
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
        help="which learning method to use",
        choices={"dqn", "ensemble_dqn"},
        required=True)

    parser.add_argument(
        "--domain", "-D",
        help="which domain to use method on",
        required=True,
        choices=["cartpole", "tiger", "gridworld"])

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
        "--horizon", "-H",
        default=200,
        type=int,
        help="length of the problem")

    parser.add_argument(
        "--episodes",
        default=100000,
        type=int,
        help="number of episodes to run")

    parser.add_argument(
        "--gamma",
        default=0.99,
        type=float,
        help="discount factor to be used")

    parser.add_argument(
        "--learning_rate", "--alpha",
        default=1e-4,
        type=float,
        help="learning rate of the policy gradient descent")

    parser.add_argument(
        "--loss",
        default="rmse",
        help="type of loss to consider",
        choices={"rmse", "huber"})

    parser.add_argument(
        "--clipping",
        action='store_true',
        help="whether or not to use clipping in computing loss")

    parser.add_argument(
        "--domain_size",
        type=int,
        default=0,
        help="size of domain (gridworld is size of grid)")

    parser.add_argument(
        "--network_size",
        required=True,
        help='the size of the q-network',
        choices=["small", "med", "large"])

    parser.add_argument(
        "--observation_len",
        default=1,
        type=int,
        help="number of past observations to provide to the policy")

    parser.add_argument(
        "--num_nets",
        default=1,
        type=int,
        help='number of nets to use in ensemble methods'
        )

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
        default=256,
        type=int,
        help="how often the target network should be updated (every # time steps)")

    parser.add_argument(
        "--train_frequency",
        default=4,
        type=int,
        help="how often the agent performs a batch update (every # time steps)")

    return parser.parse_args()

def get_environment(conf):
    """ returns environments as indicated in conf """
    if conf.domain == "tiger":
        return tiger.Tiger(conf)
    if conf.domain == "cartpole":
        return cartpole.Cartpole(conf)
    if conf.domain == "gridworld":
        return gridworld.GridWorld(conf)

def get_agent(conf, env, sess, name):
    """ returns agent given configurations and environment """
    if conf.method == "dqn":
        return DQN(env, conf, sess, name=name)
    if conf.method == "ensemble_dqn":
        return ensemble_DQN(env, conf, sess, name=name)

if __name__ == '__main__':
    main()
