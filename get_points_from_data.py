import numpy as np
import pandas as pd
from typing import List, Tuple
import os
from rocket_classes import *

radars_location = {
    'Ashdod' : Point(31.77757586390034, 34.65751251836753, 0),
    'Kiryat' : Point(31.602089287486198, 34.74535762921831, 0),
    'Ofakim': Point(31.302709659709315 ,34.59685294800365, 0),
    'Tseelim': Point(31.20184656499955, 34.52669152933695, 0),
    'Meron': Point(33.00023023451869, 35.404698698883585, 0),
    'YABA': Point(30.653610411909529 ,34.783379139342955, 0),
    'Modiin': Point(31.891980958022323, 34.99481765229601, 0),
    'Gosh': Point(32.105913486777084, 34.78624983651992, 0),
    'Carmel': Point(32.65365306190331, 35.03028065430696, 0),
}

def update_rockets(data: pd.DataFrame, rockets_list: List[Rocket]) -> None:
    nums_rockets = num_of_rockets_in_data(data)
    if nums_rockets > len(rockets_list) - 1: # if this file has more rockets add new rockets
        for i in range(len(rockets_list), nums_rockets + 1):
            rockets_list.append(Rocket(i))
    for index, row in data.iterrows(): #add new Radar point to each
        rp = RadarPoint(row["elevation"], row["azimuth"], row["range"], radars_location[data.name])
        rockets_list[row["id"]].add_point(rp)



def num_of_rockets_in_data(data: pd.DataFrame) -> int:
    id_array = data_file["id"].to_numpy('some_file.csv')
    return max(id_array)

null_rocket = Rocket(0)
rockets : List[Rocket] = [null_rocket]
directory = r"C:\Users\TLP-001\Documents\Intro\Sprint-2\With ID\Impact points data"
for filename in os.listdir(directory):
    if filename.endswith(".csv"):
        data_file = pd.read_csv(filename)
        data_file.name = filename.split("_")[0]
        update_rockets(data_file, rockets)






for
