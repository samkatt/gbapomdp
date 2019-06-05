""" runs integration tests """

import unittest

from model_free import main as mf_main, parse_arguments as mf_parse_arguments
from model_based import main as mb_main, parse_arguments as mb_parse_arguments
from pouct_planning import main as pouct_main, parse_arguments as pouct_parse_arguments


class TestModelFreeAgents(unittest.TestCase):
    """ runs default model free experiments """

    @staticmethod
    def run_experiment(args):
        """ runs an experiment with args as configuration

        Adds some default arguments to the experiment

        Args:
             args: ['--arg=val', ...] the argument to run

        """

        def_args = ['--episodes=3', '--runs=2', '--horizon=10', '-v=0']

        mf_main(mf_parse_arguments(def_args + args))

    def test_domains(self):
        """ just the default arguments on all domains """

        self.run_experiment(['-D=tiger', '--episodes=50'])
        self.run_experiment(['--domain_size=3', '-D=cartpole', "--horizon=10000"])
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
        """ tests clipping, huber loss """

        self.run_experiment(['-D=tiger', '--loss=huber', '--num_nets=3'])

        self.run_experiment(['-D=cartpole', '--clipping', "--horizon=10000"])


class TestPOMCP(unittest.TestCase):
    """ runs default PO-UCT with belief experiments """

    @staticmethod
    def run_experiment(args):
        """ runs an experiment with args as configuration

        Adds some default arguments to the experiment

        Args:
             args: ['--arg=val', ...] the argument to run

        """

        def_args = ['--num_sims=16', '--num_particles=64', '--runs=2', '--horizon=5', '-v=0']

        pouct_main(pouct_parse_arguments(def_args + args))

    def test_domains(self):
        """ just the default arguments on all discrete domains """

        self.run_experiment(['-D=tiger'])
        self.run_experiment(['--domain_size=3', '-D=gridworld'])
        self.run_experiment(['--domain_size=3', '-D=collision_avoidance'])
        self.run_experiment(['--domain_size=5', '-D=chain'])

    def test_setting_sims(self):
        """ tests setting number of simulations """

        self.run_experiment(['-D=tiger', '--num_sims=10'])


class TestModelBasedRL(unittest.TestCase):
    """ tests the model_base.py entry point """

    @staticmethod
    def run_experiment(args):
        """ runs an experiment with args as configuration

        Adds some default arguments to the experiment

        Args:
             args: ['--arg=val', ...] the argument to run

        """

        def_args = ['--num_sims=16', '--num_particles=64', '--runs=2', '--horizon=5', '-v=0', '--episodes=3']

        mb_main(mb_parse_arguments(def_args + args))

    def test_learning_true_dynamics_offline(self):
        """ tests basic offline learning """

        self.run_experiment(['-D=tiger', '--learn=true_dynamics_offline'])
        self.run_experiment(['-D=collision_avoidance', '--domain_size=5', '--learn=true_dynamics_offline'])
        self.run_experiment(['-D=chain', '--domain_size=4', '--learn=true_dynamics_offline'])
        self.run_experiment(['-D=gridworld', '--domain_size=3', '--learn=true_dynamics_offline'])


if __name__ == '__main__':
    unittest.main()
