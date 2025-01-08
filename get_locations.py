import pandas as pd
from typing import Tuple,List
import math
import numpy as np

EARTH_RADIUS = 6378137
class Radar:
    elevation = 0
    def __init__(self, longitude, latitude):
        self.longitude = longitude
        self.latitude = latitude

class Point:
    def __init__(self, longitude, latitude, altitude):
        self._longitude = longitude

class RadarPoint:
    def __init__(self, elevation, azimuth, r, radar):
        self.elevation = elevation
        self.azimuth = azimuth
        self.range = r
        self.radar = radar


def lat_calculate(r: RadarPoint) -> float:
    lat_diff = math.atan((r.range*math.cos(np.radians(r.elevation))*math.cos(np.radians(r.azimuth)))/EARTH_RADIUS)
    return lat_diff + r.radar.latitude

def lon_calculate(r: RadarPoint) -> float:
    lon_diff = math.atan((r.range*math.cos(np.radians(r.elevation))*math.sin(np.radians(r.azimuth)))/EARTH_RADIUS)
    return lon_diff + r.radar.latitude


def altitude_calculate(r: RadarPoint) -> float:
    dis_from_center = math.sqrt(EARTH_RADIUS**2 + r.elevation**2 + 2*EARTH_RADIUS*r.range*math.sin(np.radians(r.elevation)))
    return dis_from_center - EARTH_RADIUS


def process_point(radar_point: RadarPoint) -> Point:
    return Point(longitude=lon_calculate(radar_point), latitude=lat_calculate(radar_point), altitude=altitude_calculate(radar_point))


class Rocket:

    _unprocessed_locations : List[RadarPoint] = []
    _processed_locations : List[Point] = []

    def __init__(self, ID):
        self._ID = ID

    def ID(self):
        return self._ID

    def add_point(self, radar_point: RadarPoint) -> None:
        self._unprocessed_locations.append(radar_point)
        self._processed_locations.append(process_point(radar_point))



