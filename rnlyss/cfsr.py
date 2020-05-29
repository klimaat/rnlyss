# -*- coding: utf-8 -*-
#
# Copyright 2019 Klimaat

from rnlyss.grid import GaussianGrid
from rnlyss.cfsv2 import CFSV2


class CFSR(CFSV2):

    # Time
    years = 1979, 2010

    # Grid  (576x1152)
    grid = GaussianGrid(shape=(576, 1152), origin=(90, 0), delta=(-1, 360 / 1152))

    # CFSR RDA dataset
    dataset = "ds093"


def main():
    pass


if __name__ == "__main__":
    main()
