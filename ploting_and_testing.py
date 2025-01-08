import pandas as pd
from typing import Tuple,List
import math
from rocket_classes import Rocket, Point
import numpy as np
import matplotlib.pyplot as plt
from get_points_from_data import *

R = 6380 * 1000


class EcefPoint:
    def __init__(self, point):
        clat = np.cos(np.radians(point.get_latitude()))
        slat = np.sin(np.radians(point.get_latitude()))
        clon = np.cos(np.radians(point.get_longitude()))
        slon = np.sin(np.radians(point.get_longitude()))
        r = R + point.get_altitude()
        self.z = r * slat
        self.x = r * clat * clon
        self.y = r * clat * slon

    def getX(self):
        return self.x

    def getY(self):
        return self.y

    def getZ(self):
        return self.z

if __name__ == '__main__':
    rocket_points = get_data()[1].get_locations()
    ecef_points = [EcefPoint(point) for point in rocket_points]
    xs = [point.getX for point in ecef_points]
    ys = [point.getY for point in ecef_points]
    zs = [point.getZ for point in ecef_points]
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    plt.show()



