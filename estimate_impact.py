import numpy as np
import matplotlib.pyplot as plt

import get_points_from_data


def createG(time, point3, point2, point1):
    # start velocities
    v1_alt = (point2.get_altitude() - point1.get_altitude()) / (point2.get_time()-point1.get_time())
    v2_alt = (point3.get_altitude() - point2.get_altitude()) / (point3.get_time()-point2.get_time())
    v1_long = (point2.get_longitude() - point1.get_longitude()) / (point2.get_time() - point1.get_time())
    v2_long = (point3.get_longitude() - point2.get_longitude()) / (point3.get_time() - point2.get_time())
    v1_lat = (point2.get_latitude() - point1.get_latitude()) / (point2.get_time() - point1.get_time())
    v2_lat = (point3.get_latitude() - point2.get_latitude()) / (point3.get_time() - point2.get_time())

    # start displacements
    alt = [point3.get_altitude()]
    long = [point3.get_longitude()]
    lat = [point3.get_latitude()]

    for n in range(0, len(time) - 1):
        # move displacements
        alt.append(alt[-1] + 0.5 * (v2_alt + v1_alt) * (time[n + 1] - time[n]) - 9.81 * (time[n + 1] - time[n])*2)
        long.append(long[-1] + 0.5 * (v2_long + v1_long) * (time[n + 1] - time[n]))
        lat.append(lat[-1] + 0.5 * (v2_lat + v1_lat) * (time[n + 1] - time[n]))

        # mov v
        v1_alt = v2_alt
        v2_alt = (alt[n] - alt[n-1]) / (time[n+1] - time[n])
        v1_lat = v2_lat
        v2_lat = (alt[n] - alt[n-1]) / (time[n+1] - time[n])
        v1_long = v2_long
        v2_long = (alt[n] - alt[n-1]) / (time[n+1] - time[n])

        # if touches ground
        if alt[n+1] < 0:
            return np.array(alt), np.array(long), np.array(lat)
    return [np.array(alt), np.array(long), np.array(lat)]


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

    alt_corrected = np.zeros(len(time))
    point_G_prediction = createG(time, point3, point2, point1)
    for j in range(len(point_G_prediction[0])):
        alt_corrected[j] = point_G_prediction[0][j]
    disp.append(alt_corrected)

avg = 0
ground = 0
for i in range(len(disp)):
    for j in range(len(disp[i])):
        if disp[i][j] < 0:
            avg = avg + (time[i] + time[i-1])/2
avg = avg/(len(disp))
print(avg)
#
plt.scatter([point1.get_time(), point3.get_time(), points[6].get_time(), points[10].get_time()],
            [point1.get_altitude(), point3.get_altitude(), points[6].get_altitude(), points[10].get_altitude()])
plt.plot(time, disp[1])
plt.legend(["Displacement", "Velocity"])
plt.xlabel("Time")
plt.show()
