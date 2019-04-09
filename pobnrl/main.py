""" run Neural RL methods on partially observable environments """

import logging
import time

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from math import sqrt
import numpy as np
import tensorflow as tf

from agents import create_agent
from environments import create_environment
from episode import run_episode
from misc import tf_session, tf_run, log_level

VERBOSE_TO_LOGGING = {
    0: 30,  # warning
    1: 20,  # info
    2: 15,  # verbose
    3: 10,  # debug
    4: 5    # spam
}


def main(conf):
    """ main: runs an agent in an environment given configurations

    Args:
         conf: configurations as namespace from `parse_arguments`

    """

    logging.basicConfig(
        level=VERBOSE_TO_LOGGING[conf.verbose],
        format="[%(asctime)s] %(levelname)s: %(message)s \t\t\t(%(name)s)",
        datefmt='%H:%M:%S'
    )

    logger = logging.getLogger(__name__)

    cur_time = time.time()
    result_mean = np.zeros(conf.episodes)
    result_var = np.zeros(conf.episodes)

    env = create_environment(
        conf.domain,
        conf.domain_size,
        conf.verbose
    )
    agent = create_agent(
        env.action_space,
        env.observation_space,
        conf
    )

    init_op = tf.global_variables_initializer()

    logger.log(
        log_level['info'],
        "Running %s experiment on %s", str(agent), str(env)
    )

    with tf_session():
        for run in range(conf.runs):

            tf_run(init_op)
            agent.reset()

            tmp_res = np.zeros(conf.episodes)

            logger.log(log_level['info'], "Starting run %d", run)
            for episode in range(conf.episodes):

                tmp_res[episode] = run_episode(env, agent, conf)

                if episode > 0 and time.time() - cur_time > 5:

                    logger.log(
                        log_level['info'],
                        "run %d episode %d: avg return: %f",
                        run, episode,
                        np.mean(tmp_res[max(0, episode - 100):episode])
                    )

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
                header=str(conf)
                + "\nreturn mean, return var, return count, return stder"
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
        default=0,
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
        "--learning_rate", "--alpha",
        default=1e-4,
        type=float,
        help="learning rate of the policy gradient descent"
    )

    parser.add_argument(
        "--loss",
        default="rmse",
        help="type of loss to consider",
        choices={"rmse", "huber"}
    )

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
        "--prior_function_scale",
        default=0,
        type=float,
        help="The scaling factor for random prior in the loss function"
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
        help='the number of hidden nodes in the q-network',
        default=64,
        type=int
    )

    parser.add_argument(
        "--history_len",
        default=1,
        type=int,
        help="number of past observations to provide to the policy"
    )

    parser.add_argument(
        "--batch_size",
        default=32,
        type=int,
        help="size of learning batch"
    )

    parser.add_argument(
        "--target_update_freq",
        default=128,
        type=int,
        help="how often the target network is updated (every # time steps)")

    parser.add_argument(
        "--train_freq",
        default=1,
        type=int,
        help="how often the agent performs a batch update (every # time steps)"
    )

    parser.add_argument(
        "--exploration",
        default=-1,
        type=float,
        help="the e-greedy exploration, will use anealling if **not** set"
    )

    parser.add_argument(
        "--random_policy",
        action='store_true',
        help="use this flag to pick the random agent controller"
    )

    return parser.parse_args(args)


if __name__ == '__main__':
    main(parse_arguments())
