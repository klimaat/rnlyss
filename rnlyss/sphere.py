# -*- coding: utf-8 -*-
#
# Copyright 2019 Klimaat
#

import numpy as np

_EARTH_RADIUS = 6371.009


def calc_distance(lat1, lon1, lat2, lon2, r=_EARTH_RADIUS):
    """
    Calculate the great circle distance in kilometres between two points
    on the earth (specified in decimal degrees) using the haversine.
    """

    # Convert decimal degrees to radians
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

    # Haversine formula
    dlon = lon2 - lon1

    dlat = lat2 - lat1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))

    # Distance is angle*radius
    return r * c


def to_cartesian(lat, lon, r=_EARTH_RADIUS):
    """
    Convert spherical coordinates to Cartesian.
    """
    lat, lon = map(np.radians, [lat, lon])
    cos = np.cos(lat) * r
    return np.squeeze(np.c_[cos * np.cos(lon), cos * np.sin(lon), np.sin(lat) * r])


def calc_bearing(lat1, lon1, lat2, lon2):
    """
    Calculate bearing in degrees from (lat1,lon1) towards (lat2, lon2)
    """
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    a = np.arctan2(
        np.sin(dlon) * np.cos(lat2),
        np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon),
    )

    return (np.degrees(a) + 360.0) % 360.0


def calc_area(lat1, lat2, lon1, lon2, r=_EARTH_RADIUS):
    """
    Calculate the area bound by lat1, lon1, lat2, lon2
    """
    dlon = np.radians(np.where(lon1 > lon2, lon2 + 360.0 - lon1, lon2 - lon1))
    return dlon * np.fabs(np.sin(np.radians(lat2)) - np.sin(np.radians(lat1))) * r ** 2


def test():
    print(calc_bearing(45, 0, 44, 0))
    print(calc_area(-90, 90, -180, 180, r=1))
    print(calc_area(-90, 0, -180, 180, r=1))
    print(calc_area(90, 0, -180, 180, r=1))
    print(calc_area(0, 90, -90, 90, r=1))


if __name__ == "__main__":
    test()
