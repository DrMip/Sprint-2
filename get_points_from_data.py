import numpy as np
import pandas as pd
from typing import List, Tuple
import os
from rocket_classes import *

radars_location = {
    'Ashdod' : Radar(31.77757586390034, 34.65751251836753),
    'Kiryat' : Radar(31.602089287486198, 34.74535762921831),
    'Ofakim': Radar(31.302709659709315 ,34.59685294800365),
    'Tseelim': Radar(31.20184656499955, 34.52669152933695),
    'Meron': Radar(33.00023023451869, 35.404698698883585),
    'YABA': Radar(30.653610411909529 ,34.783379139342955),
    'Modiin': Radar(31.891980958022323, 34.99481765229601),
    'Gosh': Radar(32.105913486777084, 34.78624983651992),
    'Carmel': Radar(32.65365306190331, 35.03028065430696),
}

def update_rockets(data: pd.DataFrame, rockets_list: List[Rocket]) -> None:
    nums_rockets = num_of_rockets_in_data(data)
    print(len(rockets_list[0].get_locations()))
    if nums_rockets > len(rockets_list) - 1: # if this file has more rockets add new rockets
        for i in range(len(rockets_list), nums_rockets + 1):
            rockets_list.append(Rocket(i))
    for index, row in data.iterrows(): #add new Radar point to each
        rp = RadarPoint(row["elevation"], row["azimuth"], row["range"], row["time"], radars_location[data.name])

        rockets_list[int(row["ID"])].add_point(rp)



def num_of_rockets_in_data(data: pd.DataFrame) -> int:
    id_array = data["ID"].to_numpy()
    return int(max(id_array))


rockets = [Rocket(0)]
directory = r"C:\Users\TLP-001\Documents\Intro\Sprint-2\With ID\Impact points data"
for filename in os.listdir(directory) :
     if filename.endswith(".csv") and filename.split("_")[0] == "Ashdod":
        data_file = pd.read_csv(fr"{directory}\{filename}")
        data_file.name = filename.split("_")[0]
        update_rockets(data_file, rockets)

print(rockets[1])

