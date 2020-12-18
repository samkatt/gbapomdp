""" plotting functionality """

import sys
from typing import List

from po_nrl.analysis.visualization import default_plot_style, plot_experiment


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
        print(
            f"{smoothing_argument} cannot be interpret as amount of smoothing (int)"
        )

    plot_experiment(files, smooth_amount=smoothing)
    default_plot_style()
    plt.show()


if __name__ == "__main__":

    assert (
        len(sys.argv) > 2
    ), "Expects at least 2 arguments: # smoothing and list fo files"

    main(sys.argv[1], sys.argv[2:])
