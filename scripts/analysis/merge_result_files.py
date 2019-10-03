#!/usr/bin/env python
""" merges multiple `result` files into a single one """

import sys

from po_nrl.analysis.merge_result_files import merge


if __name__ == '__main__':
    sys.exit(merge())
