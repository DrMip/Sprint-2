import pandas as pd
from typing import Tuple,List
import math
import numpy as np

EARTH_RADIUS = 6378137
class Radar:
    def __init__(self, latitude, longitude):
        self.longitude = longitude
        self.latitude = latitude

class Point:
    def __init__(self, longitude, latitude, altitude, time = None):
        self._longitude = longitude
        self._latitude = latitude
        self._altitude = altitude
        self._time = time
    def get_time(self):
        return self._time
    def get_longitude(self):
        return self._longitude
    def get_latitude(self):
        return self._latitude
    def get_altitude(self,):
        return self._altitude




class RadarPoint:
    def __init__(self, elevation, azimuth, r, time, radar):
        self.elevation = elevation
        self.azimuth = azimuth
        self.range = r
        self.radar = radar
        self.time = time
    def get_time(self):
        return self.time


def lat_calculate(r: RadarPoint) -> float:
    lat_diff = math.atan((r.range*math.cos(np.radians(r.elevation))*math.cos(np.radians(r.azimuth)))/EARTH_RADIUS)
    return np.degrees(lat_diff) + r.radar.latitude

def lon_calculate(r: RadarPoint) -> float:
    lon_diff = math.atan((r.range*math.cos(np.radians(r.elevation))*math.sin(np.radians(r.azimuth)))/EARTH_RADIUS)
    return np.degrees(lon_diff) + r.radar.longitude


def altitude_calculate(r: RadarPoint) -> float:
    dis_from_center = math.sqrt(EARTH_RADIUS**2 + r.range**2 + 2*EARTH_RADIUS*r.range*math.sin(np.radians(r.elevation)))
    return dis_from_center - EARTH_RADIUS


def process_point(radar_point: RadarPoint) -> Point:
    return Point(longitude=lon_calculate(radar_point), latitude=lat_calculate(radar_point), altitude=altitude_calculate(radar_point), time=radar_point.get_time())


class Rocket:

    def __init__(self, ID):
        self._unprocessed_locations: List[RadarPoint] = []
        self._processed_locations: List[Point] = []
        self._ID = ID

    def get_ID(self):
        return self._ID


    def add_point(self, radar_point: RadarPoint) -> None:
        self._unprocessed_locations.append(radar_point)
        self._processed_locations.append(process_point(radar_point))


    def get_locations(self):
        return self._processed_locations

    def __str__(self):
        string = ""
        for point in self._processed_locations:
            string += f"Time: {point.get_time()}, Longitude: {point.get_longitude()}, Latitude: {point.get_latitude()}, Altitude: {point.get_altitude()}\n"

        return string





