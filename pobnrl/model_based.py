""" Run POMCP on partially observable, known, environments """

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from typing import Optional, List, Callable
import numpy as np
import tensorflow as tf

from agents.model_based_agents import create_learning_agent
from agents.neural_networks.neural_pomdps import DynamicsModel
from domains import create_environment, EncodeType, create_prior
from domains.learned_environments import NeuralEnsemblePOMDP
from domains.learned_environments import train_from_random_policy, train_from_uniform_steps
from environments import Simulator
from episode import run_episode
from misc import POBNRLogger, set_random_seed
from tf_api import tf_session, tf_run


def main(conf) -> None:
    """ runs PO-UCT planner with a belief on given configurations

    Args:
         conf: configurations as namespace from `parse_arguments`

    """

    # global init
    POBNRLogger.set_level(POBNRLogger.LogLevel.create(conf.verbose))
    logger = POBNRLogger('model based main')
    tf.compat.v1.reset_default_graph()

    if conf.random_seed:
        set_random_seed(conf.random_seed)

    # environment and agent setup
    env = create_environment(conf.domain, conf.domain_size, EncodeType.DEFAULT)
    assert isinstance(env, Simulator)

    sim = NeuralEnsemblePOMDP(env, conf=conf)
    agent = create_learning_agent(sim, conf)
    train_method = create_train_method(env, conf)

    init_op = tf.compat.v1.global_variables_initializer()

    # results init
    result_mean = np.zeros(conf.episodes)
    ret_m2 = np.zeros(conf.episodes)
    logger.log(POBNRLogger.LogLevel.V1, f"Running {agent} experiment on {env}")

    tensorboard_name = ""

    for run in range(conf.runs):

        if conf.tensorboard_name:
            tensorboard_name = f'{conf.tensorboard_name}-{run}'

        with tf_session(conf.use_gpu, tensorboard_name):

            tf_run(init_op)
            sim.train_models(train_method)

            agent.reset()

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
        '--tensorboard_name', '--diag',
        default="",
        help="tensorboard directory name, if applicable"
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
        "--train_offline",
        choices=['on_true', 'on_prior'],
        default='on_true',
        help='which, if applicable, type of learning to use'
    )

    parser.add_argument(
        "--offline_data_sampler",
        choices=['uniform', 'random_policy'],
        default='uniform',
        help='how to sample trajectories for offline dynamics learning'
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
        default=1,
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

    parser.add_argument(
        "--random_seed", "--seed",
        default=0,
        type=int,
        help='set random seed'
    )

    return parser.parse_args(args)  # if args is "", will read cmdline


def create_train_method(env: Simulator, conf) -> Callable[[DynamicsModel], None]:
    """ creates a model training method

    This returns a function that can be called on any
    `pobnrl.agents.neural_networks.neural_pomdps.DynamicsModel` net to be
    trained

    Args:
         env: (`pobnrl.environments.Simulator`):
         conf: namesapce of program options

    RETURNS (`Callable[[DynamicsModel], None]`): a function that trains an model

    """

    # select sampling method
    if conf.offline_data_sampler == 'random_policy':
        sample_method = train_from_random_policy
    elif conf.offline_data_sampler == 'uniform':
        sample_method = train_from_uniform_steps

    # select train_on_true versus train_on_prior
    if conf.train_offline == 'on_true':
        def sim_sampler() -> Simulator:
            return env
    elif conf.train_offline == 'on_prior':
        sim_sampler \
            = create_prior(conf.domain, conf.domain_size, EncodeType.DEFAULT).sample

    def train_method(net: DynamicsModel):
        sim = sim_sampler()
        sample_method(net, sim, conf.num_pretrain_epochs, conf.batch_size)

    return train_method


if __name__ == '__main__':
    main(parse_arguments())
