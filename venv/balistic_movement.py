import numpy as np
import matplotlib.pyplot as plt

import get_points_from_data

# input: arr = [t0, a, lat, lon]
# goal: חיתוך עם ציר X
#       פונקצית התאמה עם פרמטרים
#       למצוא את הפרמטרים


A = 100
t1 = 30
tstart = 5
def integrateGraph(A, time, array):
    resArray = [A]
    for n in range(0, len(time) - 1):
        resArray.append(
            resArray[-1] + 0.5 * (array[n + 1] + array[n]) * (time[n + 1] -
                                                              time[n])
        )
        #   
    return np.array(resArray)


totalMass = 1
dryMass = 0.906
burnTime = 2.0
totalImpulse = 49.6
propellantMass = 0.064
averageThrust = totalImpulse/burnTime
massFlowRate = propellantMass/burnTime

time = np.linspace(tstart, t1, 10*(t1-tstart), False)

if tstart <  burnTime:
    index = np.where(time == burnTime)[0][0] + 1
    # index = int(np.where(time==burnTime)[0] + 1)
else:
    index = 0
thrust = np.append(np.repeat(averageThrust, index), np.repeat(0, len(time) - index))
mass = np.append(np.repeat(totalMass, index) - time[0:index] * massFlowRate, np.repeat(dryMass, len(time) - index))
acceleration = thrust/mass - 9.81
vel = integrateGraph(A, time, acceleration)



disp = integrateGraph(acceleration[0], time , vel)


print(disp)
plt.plot(time, disp)
plt.plot(time, vel)
plt.legend(["Displacement", "Velocity"])
plt.xlabel("Time")
plt.show()
