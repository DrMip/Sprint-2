import numpy as np
import matplotlib.pyplot as plt

def integrateGraph(time, array):
    resArray = [0]
    for n in range(0, len(time) - 1):
        resArray.append(
            resArray[-1] + 0.5 * (array[n + 1] + array[n]) * (time[n + 1] -
                                                              time[n])
        )
    return np.array(resArray)


totalMass = 1
dryMass = 0.906
burnTime = 2.0
totalImpulse = 49.6
propellantMass = 0.064
averageThrust = totalImpulse/burnTime
massFlowRate = propellantMass/burnTime

time = np.linspace(-1, 10, 110, False)
index = np.where(time == burnTime)[0][0] + 1
#index = int(np.where(time==burnTime)[0] + 1)
thrust = np.append(np.repeat(averageThrust, index), np.repeat(0, len(time) - index))
mass = np.append(np.repeat(totalMass, index) - time[0:index] * massFlowRate, np.repeat(dryMass, len(time) - index))
acceleration = thrust/mass - 9.81
vel = integrateGraph(time, acceleration)
disp = integrateGraph(time, vel)



plt.plot(time, disp)
plt.plot(time, vel)
plt.legend(["Displacement", "Velocity"])
plt.xlabel("Time")
plt.show()
