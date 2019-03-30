""" runs integration tests """

import unittest

from main import main, parse_arguments


class TestDefaults(unittest.TestCase):
    """ runs default experiment """

    def test_environments(self):  # pylint: disable=no-self-use
        """ just the default arguments on some environments """

        envs = ['tiger', 'cartpole', 'collision_avoidance', 'gridworld']
        def_args = ['--domain_size=3', '--episodes=3', '--runs=3']

        for env in envs:
            args = ['-D', env] + def_args
            confs = parse_arguments(args)

            main(confs)  # pylint: disable=too-many-function-args
