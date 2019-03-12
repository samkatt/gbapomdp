""" run Neural RL methods on partially observable environments """

import time

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from math import sqrt
import numpy as np

from agents import agent
from agents.networks import neural_network_misc
from agents.networks.q_functions import DQNNet, DRQNNet
from environments import cartpole, collision_avoidance, gridworld, tiger, environment
from episode import run_episode
from misc import tf_init


def main():
    """ main: tests the performance of an agent in an environment """

    conf = parse_arguments()

    cur_time = time.time()

    tf_init()

    result_mean = np.zeros(conf.episodes)
    result_var = np.zeros(conf.episodes)

    env = get_environment(conf.domain, conf.domain_size, conf.verbose)

    for run in range(conf.runs):

        print(time.ctime(), "starting run", run)

        agent = get_agent(conf, env, name='run-' + str(run))
        tmp_res = np.zeros(conf.episodes)

        for episode in range(conf.episodes):

            tmp_res[episode] = run_episode(env, agent, conf)

            if episode > 0 and time.time() - cur_time > 5:

                print(time.ctime(),
                      f"run {run}, episode {episode}: avg return:",
                      str(np.mean(tmp_res[max(0, episode - 100):episode])))

                cur_time = time.time()

        # update mean and variance
        delta = tmp_res - result_mean
        result_mean += delta / (run + 1)
        delta_2 = tmp_res - result_mean
        result_var += delta * delta_2

        # process results into rows of for each episode
        # return avg, return var, return #, return stder
        summary = np.transpose([result_mean,
                                result_var / (run + 1),
                                [run + 1] * conf.episodes,
                                np.sqrt(
                                    result_var / (run + 1))
                                / sqrt(run + 1)])

        np.savetxt(
            conf.file,
            summary,
            delimiter=', ',
            header="return mean, return var, return count, return stder")


def parse_arguments():
    """ in control of converting command line arguments to configurationis """
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
        default=0.95,
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
        help="how often the target network is updated (every # time steps)")

    parser.add_argument(
        "--train_frequency",
        default=4,
        type=int,
        help="how often the agent performs a batch update (every # time steps)"
    )

    parser.add_argument(
        "--random_policy",
        action='store_true',
        help="use this flag to pick the random agent controller"
    )

    return parser.parse_args()


def get_environment(
        domain_name: str,
        domain_size: int,
        verbose: bool) -> environment.Environment:
    """ the factory function to construct environments

    Args:
         domain_name: (`str`): determines which domain is created
         domain_size: (`int`): the size of the domain (domain dependent)
         verbose: (`bool`): whether or not to be verbose

    RETURNS (`pobnrl.environments.environment.Environment`)

    """

    if domain_name == "tiger":
        return tiger.Tiger(verbose)
    if domain_name == "cartpole":
        return cartpole.Cartpole(verbose)
    if domain_name == "gridworld":
        return gridworld.GridWorld(domain_size, verbose)
    if domain_name == "collision_avoidance":
        return collision_avoidance.CollisionAvoidance(
            domain_size, verbose)

    raise ValueError('unknown domain ' + domain_name)


def get_agent(
        conf,
        env: environment.Environment,
        name: str) -> agent.Agent:
    """ factory function to construct agents

    Args:
         conf: configuration file (see program input -h)
         env: (`pobnrl.environments.environment.Environment`): real environment
         name: (`str`): used to provide scope for tensorflow

    RETURNS (`agents.agent.Agent`)

    """

    if conf.random_policy:
        return agent.RandomAgent(env.spaces()["A"])

    # construct Q function
    if conf.recurrent:
        qfunc = DRQNNet
        arch = neural_network_misc.TwoHiddenLayerRecQNet(
            conf.network_size)
    else:
        qfunc = DQNNet
        arch = neural_network_misc.TwoHiddenLayerQNet(
            conf.network_size)

    if conf.num_nets == 1:
        return agent.BaselineAgent(
            qfunc, arch, env, conf, name=name)

    return agent.EnsembleAgent(
        qfunc, arch, env, conf, name=name)


if __name__ == '__main__':
    main()
