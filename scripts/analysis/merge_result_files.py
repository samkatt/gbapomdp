#!/usr/bin/env python
""" merges multiple `result` files into a single one """

import sys

from general_bayes_adaptive_pomdps.analysis.merge_result_files import merge


if __name__ == '__main__':
    sys.exit(merge())
