""" plotting functionality """

import sys

from typing import List
import matplotlib.pyplot as plt
import more_itertools as mitt
import numpy as np


def plot_experiment(file_names: List[str]):
    """ plot_experiment plots returns of experiment return files

    Args:
         file_name (`List[str]`): file paths

    """

    for file_name in file_names:
        returns = np.loadtxt(file_name, delimiter=',')[:, 0].tolist()
        smooth_returns = np.mean(list(mitt.windowed(returns, 100)), axis=1)

        plt.plot(smooth_returns, label=file_name)

    plt.legend()
    plt.show()


if __name__ == "__main__":
    plot_experiment(sys.argv[1:])
