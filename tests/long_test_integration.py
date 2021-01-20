""" runs integration tests """

import unittest

from typing import List

from general_bayes_adaptive_pomdps.model_based import main as mb_main
from general_bayes_adaptive_pomdps.pouct_planning import main as pouct_main


class TestPOMCP(unittest.TestCase):
    """ runs default PO-UCT with belief experiments """

    @staticmethod
    def run_experiment(args: List[str]):
        """runs an experiment with args as configuration

        Adds some default arguments to the experiment

        Args:
             args: (`List[str]`) the argument to run

        """

        def_args = [
            "--num_sims=16",
            "--num_particles=64",
            "--runs=2",
            "--horizon=5",
            "-v=0",
        ]

        pouct_main(def_args + args)

    def test_domains(self):
        """ just the default arguments on all discrete domains """

        self.run_experiment(["-D=tiger"])
        self.run_experiment(["--domain_size=3", "-D=gridworld"])
        self.run_experiment(["--domain_size=3", "-D=collision_avoidance"])
        self.run_experiment(["--domain_size=5", "-D=chain"])
        self.run_experiment(["--domain_size=3", "-D=road_racer"])

    def test_setting_sims(self):
        """ tests setting number of simulations """

        self.run_experiment(["-D=tiger", "--num_sims=10"])


class TestModelBasedRL(unittest.TestCase):
    """ tests the model_base.py entry point """

    @staticmethod
    def run_experiment(args) -> None:
        """runs an experiment with args as configuration

        Adds some default arguments to the experiment

        Args:
             args: ['--arg=val', ...] the argument to run

        """

        def_args = [
            "--num_sims=16",
            "--num_particles=64",
            "--horizon=5",
            "-v=0",
            "--episodes=2",
            "-B=importance_sampling",
        ]

        mb_main(def_args + args)

    def test_train_on_true(self):
        """ tests basic offline learning """

        self.run_experiment(
            [
                "-D=collision_avoidance",
                "--train_offline=on_true",
                "--domain_size=3",
            ]
        )

        self.run_experiment(
            [
                "-D=chain",
                "--domain_size=4",
                "--train_offline=on_true",
            ]
        )

    def test_train_on_prior(self):
        """ tests basic offline learning """

        self.run_experiment(
            [
                "-D=collision_avoidance",
                "--domain_size=3",
                "--train_offline=on_prior",
                "-B=importance_sampling",
            ]
        )

    def test_sample_uniform_data(self):
        """ tests the functionality of training on randomly sampled data """

        self.run_experiment(["-D=gridworld", "--domain_size=4"])

    def test_importance_sampling(self) -> None:
        """ tests whether importance sampling works """

        self.run_experiment(
            ["-D=gridworld", "--domain_size=3", "-B=importance_sampling"]
        )


class TestModelUpdates(unittest.TestCase):
    """ tests the augmented importance sampling methods (perturb, backprop, etc) """

    @staticmethod
    def run_experiment(args) -> None:
        """runs an experiment with args as configuration

        Adds some default arguments to the experiment

        Args:
             args: ['--arg=val', ...] the argument to run

        """

        def_args = [
            "--num_sims=16",
            "--num_particles=64",
            "--horizon=5",
            "-v=0",
            "--episodes=2",
            "-B=importance_sampling",
        ]

        mb_main(def_args + args)

    def test_perturb(self) -> None:
        """ tests the `perturb_stdev` parameter """
        self.run_experiment(["-D=tiger", "--perturb_stdev=.01"])

    def test_backprop(self) -> None:
        """ tests the backprop-during-belief-update functionality """
        self.run_experiment(["-D=gridworld", "--domain_size=3", "--backprop"])


if __name__ == "__main__":
    unittest.main()
