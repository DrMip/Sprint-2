import pandas as pd
from typing import Tuple,List
import math
class Radar:
    _elevation = 0
    def __init__(self, longitude, latitude):
        self._longitude = longitude
        self._latitude = latitude
    def longitude(self):
        return self._longitude
    def latitude(self):
        return self._latitude
    def elevation(self):
        return self._elevation
    def get_location(self):
        return self._longitude, self._latitude, self._elevation

class Point:
    def __init__(self, longitude, latitude, altitude):
        self._longitude = longitude

class RadarPoint:
    def __init__(self, elevation, azimuth, r, radar):
        self.elevation = elevation
        self.azimuth = azimuth
        self.range = r
        self.radar = radar

class Rocket:
    _unprocessed_locations : List[RadarPoint] = []
    _processed_locations : List[Point] = []

    def __init__(self, ID):
        self._ID = ID

    def ID(self):
        return self._ID

    def add_point(self, radar_point: RadarPoint) -> None:
        self._unprocessed_locations.append(radar_point)
        self._processed_locations.append(self.process_point(radar_point))


    def process_point(self, radar_point: RadarPoint) -> Point:
        pass

    def lat_calculate(self, radar_point: RadarPoint) -> float:
        lat_diff = math.atan((radar_point.range))

    def lon_calculate(self, radar_point: RadarPoint) -> float:

    def altitude_calculate(self, radar_point: RadarPoint) -> float:
