import numpy as np
import matplotlib.pyplot as plt

import get_points_from_data


def createG(time, point3, point2, point1):
    v1_alt =
    v2_alt =
    v1_long =
    v2_long =
    v1_lat =
    v2_lat =
    alt = [point3.get_altitude()]
    long = [point3.get_longitude()]
    lat = [point3.get_latitude()]
    for n in range(0, len(time) - 1):
        alt.append(alt[-1] + 0.5 * (v2 + v1) * (time[n + 1] - time[n])- 9.81* (time[n + 1] - time[n])*2)
        v1 = v2
        v2 = (alt[n] - alt[n-1]) / (time[n+1] - time[n])
        if alt[n+1] < 0:
            return np.array(alt)
    return np.array(alt)


points = get_points_from_data.get_data()[1].get_locations()
points = points[:200]
disp = []
for i in range(len(points) - 2):
    point1 = points[i]
    point2 = points[i+1]
    point3 = points[i+2]

    t_start = point3.get_time()
    t1 = point3.get_time() + 50
    time = np.linspace(t_start, t1, 500, False)

    v1 = (point2.get_altitude() - point1.get_altitude()) / (point2.get_time()-point1.get_time())
    v2 = (point3.get_altitude() - point2.get_altitude()) / (point3.get_time()-point2.get_time())
    disp_corrected = np.zeros(len(time))
    point_G_prediction = createG(time, point3, v1, v2)
    for j in range(len(point_G_prediction)):
        disp_corrected[j] = point_G_prediction[j]
    disp.append(disp_corrected)

avg = 0
ground = 0
for i in range(len(disp)):
    for j in range(len(disp[i])):
        if disp[i][j] < 0:
            avg = avg + (time[i] + time[i-1])/2
avg = avg/(len(disp))
print(avg)

plt.plot(time - 1736300020, disp[0])
plt.legend(["Displacement", "Velocity"])
plt.xlabel("Time")
plt.show()
