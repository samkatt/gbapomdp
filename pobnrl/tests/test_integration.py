""" runs integration tests """

import unittest

from model_free import main as mf_main, parse_arguments as mf_parse_arguments
from pomcp import main as pomcp_main, parse_arguments as pomcp_parse_arguments


class TestModelFreeAgents(unittest.TestCase):
    """ runs default model free experiments """

    @staticmethod
    def run_experiment(args):
        """ runs an experiment with args as configuration

        Adds some default arguments to the experiment

        Args:
             args: ['--arg=val', ...] the argument to run

        """

        def_args = ['--episodes=3', '--runs=2', '--horizon=10']

        # pylint: disable=too-many-function-args
        mf_main(mf_parse_arguments(def_args + args))

    def test_environments(self):  # pylint: disable=no-self-use
        """ just the default arguments on all environments """

        self.run_experiment(['-D=tiger'])
        self.run_experiment(
            ['--domain_size=3', '-D=cartpole', "--horizon=10000"])
        self.run_experiment(['--domain_size=3', '-D=gridworld'])
        self.run_experiment(['--domain_size=3', '-D=collision_avoidance'])
        self.run_experiment(['--domain_size=5', '-D=chain'])

    def test_ensemble(self):
        """ run the ensemble agent """
        self.run_experiment(['-D=tiger', '--num_nets=3'])
        self.run_experiment(
            ['-D=tiger', '--num_nets=2', '--prior_function_scale=2']
        )

        self.run_experiment(
            ["-D=gridworld", "--domain_size=3",
             "--recurrent", "--prior_function_scale=1"]
        )

        self.run_experiment(
            ["-D=collision_avoidance", "--domain_size=3",
             "--recurrent", "--prior_function_scale=2.5", "--exploration=.1"]
        )

    def test_recurrent(self):
        """ tests whether recurrent network runs correctly """

        # baseline agent
        self.run_experiment(
            ['-D=collision_avoidance',
             '--domain_size=5',
             '--recurrent',
             '--history_len=3']
        )

        # ensemble agent
        self.run_experiment(
            ['-D=gridworld',
             '--domain_size=3',
             '--recurrent',
             '--history_len=3',
             '--num_nets=2']
        )

    def test_basic_features(self):
        """ tests clipping, double_q, huber loss """

        self.run_experiment(['-D=tiger', '--loss=huber', '--num_nets=3'])

        self.run_experiment(
            ['-D=cartpole', '--clipping', "--horizon=10000"]
        )

        self.run_experiment(
            ['--domain_size=5',
             '-D=collision_avoidance',
             '--double_q',
             '--recurrent']
        )


class TestPOMCPAgents(unittest.TestCase):
    """ runs default PO-UCT with belief experiments """

    @staticmethod
    def run_experiment(args):
        """ runs an experiment with args as configuration

        Adds some default arguments to the experiment

        Args:
             args: ['--arg=val', ...] the argument to run

        """

        def_args = ['--runs=2', '--horizon=5']

        # pylint: disable=too-many-function-args
        pomcp_main(pomcp_parse_arguments(def_args + args))

    def test_environments(self):  # pylint: disable=no-self-use
        """ just the default arguments on all discrete environments """

        self.run_experiment(['-D=tiger'])
        self.run_experiment(['--domain_size=3', '-D=gridworld'])
        self.run_experiment(['--domain_size=3', '-D=collision_avoidance'])
        self.run_experiment(['--domain_size=5', '-D=chain'])


if __name__ == '__main__':
    unittest.main()
