""" run Neural RL methods on partially observable environments """

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import time
from math import sqrt
import numpy as np

from agents import agent
from agents.baseline_agent import BaselineAgent
from agents.ensemble_agent import EnsembleAgent
from agents.networks import architectures as archs
from agents.networks.qnets import DQNNet
from agents.networks.qnets import DRQNNet
from environments import cartpole
from environments import collision_avoidance
from environments import environment
from environments import gridworld
from environments import tiger
from episode import run_episode
from utils import tf_wrapper


def main():
    """ start of program """

    conf = parse_arguments()

    cur_time = time.time()
    result_mean = np.zeros(conf.episodes)
    result_var = np.zeros(conf.episodes)

    tf_wrapper.init()

    env = get_environment(conf)

    for run in range(conf.runs):

        print(time.ctime(), "starting run", run)

        agent = get_agent(conf, env, name='run-' + str(run))
        tmp_res = np.zeros(conf.episodes)

        for episode in range(conf.episodes):

            tmp_res[episode] = run_episode(env, agent, conf)

            if episode > 0 and time.time() - cur_time > 5:

                print(time.ctime(),
                      "run", run, "episode", episode,
                      ": avg return",
                      np.mean(tmp_res[max(0, episode - 100):episode]))

                cur_time = time.time()

        # update mean and variance
        delta = tmp_res - result_mean
        result_mean += delta / (run + 1)
        delta_2 = tmp_res - result_mean
        result_var += delta * delta_2

        # process results into rows of for each episode
        # return avg, return var, return #, return stder
        summary = np.transpose([
            result_mean,
            result_var,
            [run + 1] * conf.episodes,
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
        "--domain", "-D",
        help="which domain to use method on",
        required=True,
        choices=["cartpole", "tiger", "gridworld", "collision_avoidance"])

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
        default="huber",
        help="type of loss to consider",
        choices={"rmse", "huber"})

    parser.add_argument(
        "--clipping",
        action='store_true',
        help="whether or not to use clipping in computing loss"
    )

    parser.add_argument(
        "--domain_size",
        type=int,
        default=0,
        help="size of domain (gridworld is size of grid)"
    )

    parser.add_argument(
        "--num_nets",
        default=1,
        type=int,
        help='number of nets to use in ensemble methods (assumes ensemble)'
    )

    parser.add_argument(
        "--random_priors",
        action='store_true',
        help="will incorporate random prior in loss function"
    )

    parser.add_argument(
        "--recurrent",
        action='store_true',
        help="whether to use recurrent networks"
    )

    parser.add_argument(
        "--double_q",
        action='store_true',
        help="whether to use doubleQ technique"
    )

    parser.add_argument(
        "--network_size",
        required=True,
        help='the size of the q-network',
        choices=["small", "med", "large"]
    )

    parser.add_argument(
        "--observation_len",
        default=1,
        type=int,
        help="number of past observations to provide to the policy"
    )

    parser.add_argument(
        "--replay_buffer_size", "--rb_size",
        default=1000000,
        type=int,
        help="size of replay buffer"
    )

    parser.add_argument(
        "--batch_size",
        default=32,
        type=int,
        help="size of learning batch"
    )

    parser.add_argument(
        "--q_target_update_freq",
        default=256,
        type=int,
        help="how often the target network should be updated (every # time steps)")

    parser.add_argument(
        "--train_frequency",
        default=4,
        type=int,
        help="how often the agent performs a batch update (every # time steps)"
    )

    return parser.parse_args()


def get_environment(conf) -> environment.Environment:
    """get_environment returns environments as indicated in conf

    FIXME: raise exception if not correct domain

    :param conf: the conigurations given to program call parser.parse_arguments()
    :rtype: environment.Environment

    """

    if conf.domain == "tiger":
        return tiger.Tiger(conf.verbose)
    if conf.domain == "cartpole":
        return cartpole.Cartpole(conf.verbose)
    if conf.domain == "gridworld":
        return gridworld.GridWorld(conf.domain_size, conf.verbose)
    if conf.domain == "collision_avoidance":
        return collision_avoidance.CollisionAvoidance(conf.domain_size, conf.verbose)

    raise ValueError('unknown domain ' + conf.domain)


def get_agent(conf, env, name) -> agent.Agent:
    """get_agent returns agent given configurations and environment

    :param conf: configuration object from parser.parse_args()
    :param env: environment, used for info such as the observation space
    :param name: name of the agent

    :rtype: agent.Agent
    """

    # construct Q function
    if conf.recurrent:
        qfunc = DRQNNet.DRQNNet
        arch = archs.TwoHiddenLayerRecQNet(conf.network_size)
    else:
        qfunc = DQNNet.DQNNet
        arch = archs.TwoHiddenLayerQNet(conf.network_size)

    if conf.num_nets == 1:
        return BaselineAgent(qfunc, arch, env, conf, name=name)

    return EnsembleAgent(qfunc, arch, env, conf, name=name)


if __name__ == '__main__':
    main()
