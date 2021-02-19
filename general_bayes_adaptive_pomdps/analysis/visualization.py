"""lib functions to enable plotting more easily"""
from typing import List, Optional

import matplotlib.pyplot as plt
import more_itertools as mitt
import numpy as np


def default_plot_style() -> None:
    """Default function to show a plot

    Assumes `legend()` is called already, because this is such a common thing
    done by the caller
    """
    plt.xlabel("episodes")
    plt.ylabel("average reward")
    plt.tight_layout(pad=0)


def plot_experiment(
    file_names: List[str],
    smooth_amount: int = 25,
    colors: Optional[List[str]] = None,
    labels: Optional[List[str]] = None,
) -> None:
    """adds lines from data in `file_names`

    default colors used:
    ['#ff7f0e', '#1f77b4', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

    Args:
         file_name (`List[str]`): file paths
         smooth_amount (`int`): amount of smoothing to apply
         colors (`Optional[List[str]]`): optional list of colors to use for plotting

    """

    if labels is None:
        labels = file_names

    for i in range(len(file_names)):
        returns = np.loadtxt(file_names[i], delimiter=",")[:, 0].tolist()
        stder = np.loadtxt(file_names[i], delimiter=",")[:, 3].tolist()

        returns = np.mean(list(mitt.windowed(returns, smooth_amount)), axis=1)
        stder = stder[: len(returns)]

        if colors:
            plt.plot(returns, label=labels[i], color=colors[i])
            plt.fill_between(
                range(returns.shape[0]),
                returns - stder,
                returns + stder,
                alpha=0.3,
                color=colors[i],
            )
        else:
            plt.plot(returns, label=labels[i])
            plt.fill_between(
                range(returns.shape[0]),
                returns - stder,
                returns + stder,
                alpha=0.3,
                color=colors[i],
            )
