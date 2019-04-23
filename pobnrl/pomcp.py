""" Run POMCP on partiall ymethods on partially observable environments """

import copy

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import numpy as np

from environments import create_environment
from agents import create_agent
from episode import run_episode
from misc import POBNRLogger, LogLevel


def main(conf):
    """ runs PO-UCT planner with a belief on given configurations

    Args:
         conf: configurations as namespace from `parse_arguments`

    """

    POBNRLogger.set_level(LogLevel.create(conf.verbose))
    logger = POBNRLogger('pomcp main')

    ret_mean = ret_var = 0

    env = create_environment(
        conf.domain,
        conf.domain_size,
        conf.verbose
    )

    sim = copy.deepcopy(env)

    conf.agent_type = "planning"
    agent = create_agent(sim, conf)

    logger.log(LogLevel.V1, f"Running {agent} experiment on {env}")

    for run in range(conf.runs):

        ret = run_episode(env, agent, conf)

        # update mean and variance
        delta = ret - ret_mean
        ret_mean += delta / (run + 1)
        delta_2 = ret - ret_mean
        ret_var += delta * delta_2

        logger.log(LogLevel.V1, f"run {run}, avg return {ret_mean}")

        np.savetxt(
            conf.file,
            [
                ret_mean,
                ret_var,
                run + 1,
                np.sqrt(ret_var / (run + 1)) / np.sqrt(run + 1)
            ],
            delimiter=', ',
            header=str(conf) +
            "\nreturn mean, return var, return count, return stder"
        )


def parse_arguments(args: str = None):
    """ converges arguments from commandline (or string) to namespace

    Args:
         args: (`str`): a string of arguments, uses cmdline if None

    """
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        "--verbose", "-v",
        choices=[0, 1, 2, 3, 4, 5],
        default=1,
        type=int,
        help="level of logging")

    parser.add_argument(
        "--domain", "-D",
        help="which domain to use method on",
        required=True,
        choices=[
            "cartpole", "tiger", "gridworld", "collision_avoidance", "chain"
        ]
    )

    parser.add_argument(
        "--domain_size",
        type=int,
        default=0,
        help="size of domain (gridworld is size of grid)"
    )

    parser.add_argument(
        "--file", "-f",
        default="results.npy",
        help="output file path"
    )

    parser.add_argument(
        "--runs",
        default=1,
        type=int,
        help="number of runs to average returns over"
    )

    parser.add_argument(
        "--horizon", "-H",
        default=1000,
        type=int,
        help="length of the problem"
    )

    parser.add_argument(
        "--gamma",
        default=0.95,
        type=float,
        help="discount factor to be used"
    )

    parser.add_argument(
        "--random_policy",
        action='store_true',
        help="use this flag to pick the random agent controller"
    )

    parser.add_argument(
        "--num_sims",
        default=512,
        help="number of simulations/iterations to run per step"
    )

    parser.add_argument(
        "--exploration",
        type=float,
        default=1,
        help="PO-UCT (UCB) exploration constant"
    )

    parser.add_argument(
        "--belief", "-B",
        default="rejection_sampling",
        help="type of belief update",
        choices=['rejection_sampling', 'importance_sampling']
    )

    parser.add_argument(
        "--num_particles",
        default=512,
        help='number of particles in belief',
        type=int
    )

    return parser.parse_args(args)  # if args == None, will read cmdline


if __name__ == '__main__':
    main(parse_arguments())
