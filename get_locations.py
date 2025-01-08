import pandas as pd
class radar:
    def __init__(self, longitude, latitude):
        self._longitude = longitude
        self._latitude = latitude
    def longitude(self):
        return self._longitude
    def latitude(self):
        return self._latitude


class Rocket:
    _unprocessed_locations = []
    _processed_locations = []

    def __init__(self, ID):
        self._ID = ID

    def ID(self):
        return self._ID

    def add_point(self, range, elevation, azimuth, ):
        self._unprocessed_locations.append((range, elevation, azimuth))
        self._processed_locations.append(self.process_point((range, elevation, azimuth)))


    def process_point(self, range, elevation, azimuth):
        pass