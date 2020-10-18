""" Run POMCP on partially observable, known, environments """

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from math import sqrt
from typing import List, Optional
import numpy as np

from po_nrl.agents.model_based_agents import create_planning_agent
from po_nrl.domains import create_environment, EncodeType
from po_nrl.environments import Simulator
from po_nrl.episode import run_episode
from po_nrl.misc import POBNRLogger, set_random_seed


def main(args: Optional[List[str]]) -> None:
    """ runs PO-UCT planner with a belief on given configurations

    Args:
         args: (`Optional[List[str]]`): optional list of arguments

    RETURNS (`None`):

    """

    conf = parse_arguments(args)

    POBNRLogger.set_level(POBNRLogger.LogLevel.create(conf.verbose))
    logger = POBNRLogger('model based main')

    if conf.random_seed:
        set_random_seed(conf.random_seed)

    ret_mean = ret_m2 = .0

    env = create_environment(
        conf.domain,
        conf.domain_size,
        EncodeType.DEFAULT
    )

    sim = env
    assert isinstance(sim, Simulator)

    agent = create_planning_agent(sim, conf)

    logger.log(POBNRLogger.LogLevel.V1, f"Running {agent} experiment on {env}")

    agent.reset()
    for run in range(conf.runs):

        env.reset()
        agent.reset()

        ret = run_episode(env, agent, conf)

        # update mean and variance
        delta = ret - ret_mean
        ret_mean += delta / (run + 1)
        delta_2 = ret - ret_mean
        ret_m2 += delta * delta_2

        logger.log(POBNRLogger.LogLevel.V1, f"run {run}, avg return {ret_mean}")

        ret_var = 0 if run < 2 else ret_m2 / (run - 1)
        stder = 0 if run < 2 else sqrt(ret_var / run)

        np.savetxt(
            conf.file,
            [[
                ret_mean,
                ret_var,
                run + 1,
                stder
            ]],
            delimiter=', ',
            header=f"{conf}\nreturn mean, return var, return count, return stder"
        )


def parse_arguments(args: Optional[List[str]] = None):
    """ converges arguments from commandline (or string) to namespace

    Args:
         args: (`Optional[List[str]]`): a string of arguments, uses cmdline if None

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
            "tiger", "gridworld", "collision_avoidance", "chain", "road_racer"
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
        "--num_sims",
        default=512,
        type=int,
        help="number of simulations/iterations to run per step"
    )

    parser.add_argument(
        "--exploration",
        type=float,
        default=1,
        help="PO-UCT (UCB) exploration constant"
    )

    parser.add_argument(
        "--rollout_policy",
        type=str,
        choices=["", "default", "gridverse-extra"],
        default="",
        help="Rollout policy description; currently only applicable to gridverse,\
                which accepts 'gridverse-extra' for the extra-good rollout",
    )

    parser.add_argument(
        "--search_depth", "-d",
        type=int,
        default=0,
        help="The max depth of the MCTS search tree, if not set will be horizon"
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

    parser.add_argument(
        "--random_seed", "--seed",
        default=0,
        type=int,
        help='set random seed'
    )

    parsed_args = parser.parse_args(args)

    if not parsed_args.search_depth:
        parsed_args.search_depth = parsed_args.horizon

    return parsed_args
