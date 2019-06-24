""" merges multiple `result` files into a single one """

# pylint: disable=invalid-name

import sys

from typing import List, Dict
import numpy as np

sys.setrecursionlimit(10000)


def main() -> int:
    """ main """

    if len(sys.argv) <= 2:
        print("Requires at least two files to merge")
        return 0

    # read in files
    input_content: List[np.ndarray] \
        = [np.loadtxt(input_file, delimiter=",") for input_file in sys.argv[1:]]
    file_statistics = extract_statistics(input_content)

    # combine statistics
    combined_stats = combine_var_and_mean(file_statistics)
    combined_stats['stder'] = np.sqrt(combined_stats['var'] / combined_stats['n'])

    print_stats(combined_stats)

    return 1


def extract_statistics(input_content: List[np.ndarray]) -> List[Dict[str, np.ndarray]]:
    """ extracts statistics from `input_content`

    Args:
         input_content: (`List[np.ndarray]`): list of file contents

    RETURNS (`List[dict], np.array`): Where the dictionary contains {'mu', 'var', 'n'}
    as the mean, variance and number of runs

    """

    return_mean_index = 0
    return_var_index = 1
    return_count_index = 2

    statistics = []

    contains_planning_result = contains_learning_result = False

    for content in input_content:

        if len(content.shape) == 1:  # planning results

            contains_planning_result = True
            assert not contains_learning_result, f'found both plannign and learning results'

            statistics.append({
                'mu': content[return_mean_index],
                'var': content[return_var_index],
                'n': content[return_count_index],
            })

        else:  # bapomdp file

            assert len(content.shape) == 2, \
                f'unknown size content {content.shape}, expected 2-dimensional'

            contains_learning_result = True
            assert not contains_planning_result, f'found both plannign and learning results'

            statistics.append({
                'mu': content[:, return_mean_index],
                'var': content[:, return_var_index],
                'n': content[:, return_count_index],
            })

    return statistics


def combine_var_and_mean(statistics: List[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
    """ takes mean and variance of multiple files and combines in 1

    Args:
         statistics: (`List[dict]`): [{'mu', 'var', 'n'}, ....]

    RETURNS (`dict`): a single aggregated dictionary

    """
    # statistics = [{mu,var,n} ... ]
    # out: combined {mu, var, n}

    stat1 = statistics[0]
    # recursion here
    stat2 = statistics[1] if (
        len(statistics) == 2) else combine_var_and_mean(statistics[1:])

    n = stat1['n'] + stat2['n']
    mu = (stat1['mu'] * stat1['n'] + stat2['mu'] * stat2['n']) / n

    var = real2sample(
        (
            stat1['n'] * (sample2real(stat1['var'], stat1['n']) + stat1['mu']**2)
            + stat2['n'] * (sample2real(stat2['var'], stat2['n']) + stat2['mu']**2)  # noqa W503
        ) / n - mu**2, n
    )

    return {'mu': mu, 'var': var, 'n': n}


def print_stats(combined_stats: Dict[str, np.ndarray]) -> None:
    """ prints statistics in `combined_stats`

    Args:
         combined_stats: (`Dict[str, np.ndarray]`): [{'mu','var','n'}...]

    RETURNS (`None`):

    """

    print("# version 1:\n# return mean, return var, return count, return stder")

    if combined_stats['mu'].shape == ():  # planning results
        print(f"{combined_stats['mu']}, "
              f"{combined_stats['var']}, "
              f"{combined_stats['n']}, "
              f"{combined_stats['stder']}"
              )

    else:  # bapomdp results
        for i in range(len(combined_stats['mu'])):
            print(f"{combined_stats['mu'][i]}, "
                  f"{combined_stats['var'][i]}, "
                  f"{combined_stats['n'][i]}, "
                  f"{combined_stats['stder'][i]}"
                  )


def sample2real(v: float, n: int) -> float:
    """ transforms sample variance to real variance

    Args:
         v: (`float`): sample variance of population
         n: (`int`): size of population

    RETURNS (`float`): real variance of population

    """
    return ((n - 1) / n) * v


def real2sample(v: float, n: int) -> float:
    """ transforms real variance to sample variance

    Args:
         v: (`float`): real variance of population
         n: (`int`): size of population

    RETURNS (`float`): sample variance of population

    """
    return (n / (n - 1)) * v


if __name__ == '__main__':
    sys.exit(main())
