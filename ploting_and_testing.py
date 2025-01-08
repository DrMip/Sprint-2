import pandas as pd
from typing import Tuple,List
import math
from get_locations import Rocket
import numpy as np

class main:
    _rocket = None
    def __init__(self, rocket):
        self._rocket = rocket
    