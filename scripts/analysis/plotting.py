""" plotting functionality """

import sys

from typing import List
import matplotlib
import matplotlib.pyplot as plt
import more_itertools as mitt
import numpy as np


# matplotlib.rc('font', size=20)

def main(smoothing_argument: str, files: List[str]) -> None:
    """ main: validates user input and plots files

    Args:
         smoothing_argument: (`str`): amount of smoothing in the plots
         *files: list of files

    RETURNS (`None`):

    """

    try:
        smoothing = int(smoothing_argument)
    except ValueError:
        print(f"{smoothing_argument} cannot be interpret as amount of smoothing (int)")

    plot_experiment(files, smooth_amount=smoothing)


def plot_experiment(file_names: List[str], smooth_amount: int = 25) -> None:
    """ plot_experiment plots returns of experiment return files

    Args:
         file_name (`List[str]`): file paths

    """

    #   colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

    for i, file_name in enumerate(file_names):
        returns = np.loadtxt(file_name, delimiter=',')[:, 0].tolist()

        returns = np.mean(list(mitt.windowed(returns, smooth_amount)), axis=1)

        plt.plot(returns, label=file_name)
        # plt.plot(returns, label=file_name, color=colors[i])

    plt.tight_layout()
    plt.legend()
    plt.show()


if __name__ == "__main__":

    assert len(sys.argv) > 2, "Expects at least 2 arguments: # smoothing and list fo files"

    main(sys.argv[1], sys.argv[2:])
