""" Run POMCP on partially observable, known, environments """

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from typing import Optional, List
import numpy as np
import tensorflow as tf

from agents.model_based_agents import create_learning_agent
from domains import create_environment, EncodeType
from domains.learned_environments import NeuralEnsemble
from episode import run_episode
from misc import POBNRLogger, tf_session, tf_run


def main(conf) -> None:
    """ runs PO-UCT planner with a belief on given configurations

    Args:
         conf: configurations as namespace from `parse_arguments`

    """

    assert conf.learn == 'true_dynamics_offline', \
        "please set --learn flag correctly"

    POBNRLogger.set_level(POBNRLogger.LogLevel.create(conf.verbose))
    logger = POBNRLogger('model based main')

    result_mean = np.zeros(conf.episodes)
    ret_m2 = np.zeros(conf.episodes)

    env = create_environment(
        conf.domain,
        conf.domain_size,
        EncodeType.DEFAULT
    )

    sim = NeuralEnsemble(env, conf=conf, name="ensemble_pomdp")

    agent = create_learning_agent(sim, conf)

    init_op = tf.global_variables_initializer()

    logger.log(POBNRLogger.LogLevel.V1, f"Running {agent} experiment on {env}")

    with tf_session(conf.use_gpu):
        for run in range(conf.runs):

            tf_run(init_op)

            agent.reset()

            if conf.learn == 'true_dynamics_offline':
                sim.learn_dynamics_offline(env, conf.num_pretrain_epochs)

            tmp_res = np.zeros(conf.episodes)

            logger.log(POBNRLogger.LogLevel.V1, f"Starting run {run}")
            for episode in range(conf.episodes):

                env.reset()
                agent.episode_reset(None)

                tmp_res[episode] = run_episode(env, agent, conf)

                logger.log(
                    POBNRLogger.LogLevel.V1,
                    f"run {run} episode {episode}: avg return: {np.mean(tmp_res[max(0, episode - 100):episode+1])}"
                )

            # update mean and variance
            delta = tmp_res - result_mean
            result_mean += delta / (run + 1)
            delta_2 = tmp_res - result_mean
            ret_m2 += delta * delta_2

            ret_var = np.zeros(conf.episodes) if run < 2 else ret_m2 / (run - 1)
            stder = np.zeros(conf.episodes) if run < 2 else np.sqrt(ret_var / run)

            # process results into rows of for each episode
            # return avg, return var, return #, return stder
            summary = np.transpose([
                result_mean,
                ret_var,
                [run + 1] * conf.episodes,
                stder
            ])

            np.savetxt(
                conf.file,
                summary,
                delimiter=', ',
                header=f"{conf}\nreturn mean, return var, return count, return stder"
            )


def parse_arguments(args: Optional[List[str]] = None):
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
        "--episodes",
        default=1000,
        type=int,
        help="number of episodes to run"
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
        "--learn",
        choices=['true_dynamics_offline'],
        default='true_dynamics_offline',
        help='which, if applicable, type of learning to use'
    )

    parser.add_argument(
        "--learning_rate", "--alpha",
        default=1e-4,
        type=float,
        help="learning rate of the policy gradient descent"
    )

    parser.add_argument(
        "--network_size",
        help='the number of hidden nodes in the q-network',
        default=32,
        type=int
    )

    parser.add_argument(
        "--num_nets",
        default=4,
        type=int,
        help='number of learned dynamic models'
    )

    parser.add_argument(
        "--batch_size",
        default=32,
        type=int,
        help="size of learning batch"
    )

    parser.add_argument(
        "--num_pretrain_epochs",
        default=100,
        type=int,
        help="number of batch training offline"
    )

    parser.add_argument(
        "--use_gpu",
        action='store_true',
        help='enables gpu usage'
    )

    return parser.parse_args(args)  # if args is "", will read cmdline


if __name__ == '__main__':
    main(parse_arguments())
