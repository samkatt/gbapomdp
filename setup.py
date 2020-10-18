""" installation script of po_nrl """

from setuptools import find_packages, setup

setup(
    name='po_nrl',
    version='0.1.0',
    packages=find_packages(),
    test_suite='tests',
    scripts=[
        'scripts/experiments/model_based.py',
        'scripts/experiments/pouct_planning.py',
        'scripts/experiments/sequential_distr_learning.py',
        'scripts/analysis/merge_result_files.py',
        'scripts/analysis/plotting.py'
    ]
)
