#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright 2019 Klimaat

import argparse
import rnlyss.dataset


def main():
    """
    Helper to stack datasets into HDF hyperslab
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
        metavar="year",
        nargs="+",
        help="Specify year(s); default is to include all available",
    )

    parser.add_argument(
        "-m",
        "--months",
        type=int,
        metavar="month",
        nargs="+",
        help="Specify month(s); default is to include all available",
    )

    parser.add_argument(
        "--hof", action="store_true", help="Download Handbook variables"
    )

    parser.add_argument("-f", "--force", action="store_true", help="Force overwrite.")

    parser.add_argument(
        "-l", "--list", action="store_true", help="List years stacked then quit"
    )

    args = parser.parse_args()

    # Get dataset module
    dset = rnlyss.dataset.load_dataset(args.dset)

    # Ensure Handbook required variables are in args.dvars
    if args.hof:
        dvars = [dset.get_dvar(_) for _ in ["tas", "tdps", "huss", "ps", "uas", "vas"]]
        if args.dvars:
            dvars += args.dvars
        args.dvars = sorted(list({_ for _ in dvars if _ is not None}))

    if args.list:
        dates = dset.get_stacked_dates(args.dvars, args.years)
        for dvar, df in dates.items():
            print(dvar)
            print(df.astype(int))
        return

    # Instance associated stacker
    dset.stack(**vars(args))


if __name__ == "__main__":
    main()
