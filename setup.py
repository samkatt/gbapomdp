""" installation script of po_nrl """

from setuptools import find_packages, setup

requirements = [
    "pomdp_belief_tracking @ git+ssh://git@github.com/samkatt/pomdp-belief-tracking.git@main",
    "online_pomdp_planning @ git+ssh://git@github.com/samkatt/online-pomdp-planning.git@master",
    "gym_gridverse @ git+ssh://git@github.com/abaisero/gym-gridverse.git@master",
    "opencv-python",
    "torch",
    "typing_extensions",
    "pdoc3",
    "matplotlib",
    "more_itertools",
    "tensorboard",
    "mypy_extensions",
    "numpy",
]

setup(
    name='po_nrl',
    version='0.1.0',
    packages=find_packages(),
    test_suite='tests',
    install_requires=requirements,
    scripts=[
        'scripts/experiments/model_based.py',
        'scripts/experiments/pouct_planning.py',
        'scripts/experiments/sequential_distr_learning.py',
        'scripts/analysis/merge_result_files.py',
        'scripts/analysis/plotting.py'
    ]
)
