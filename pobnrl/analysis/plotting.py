""" plotting functionality """

import sys

import matplotlib.pyplot as plt
import more_itertools as mitt
import numpy as np


def plot_experiment(file_name: str):
    """ plot_experiment plots a single line given file

    Args:
         file_name (`str`): file path

    """
    returns = np.loadtxt(file_name, delimiter=',')[:, 0].tolist()
    smooth_returns = np.mean(list(mitt.windowed(returns, 10)), axis=1)
    print(smooth_returns)

    plt.plot(smooth_returns)
    plt.show()


if __name__ == "__main__":
    plot_experiment(sys.argv[1])
