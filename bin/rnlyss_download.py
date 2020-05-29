#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright 2019 Klimaat

import argparse
import rnlyss.dataset


def main():
    """
    Helper to download datasets
    """

    parser = argparse.ArgumentParser(
        description=main.__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("dset", help="Specify dataset")

    parser.add_argument(
        "-v",
        "--dvars",
        metavar="str",
        nargs="+",
        help="Specify dataset variable; default is to include all available",
    )

    parser.add_argument(
        "-y",
        "--years",
        type=int,
        metavar="int",
        nargs="+",
        help="Specify year(s); default is to include all available",
    )

    parser.add_argument(
        "-m",
        "--months",
        type=int,
        metavar="int",
        nargs="+",
        help="Specify month(s); default is to include all available",
    )

    parser.add_argument(
        "-i", "--ignore", action="store_true", help="Ignore date and file size check"
    )

    args = parser.parse_args()

    # Get dataset module
    dset = rnlyss.dataset.load_dataset(args.dset)

    # Instance associated stacker
    dset.download(**vars(args))


if __name__ == "__main__":
    main()
