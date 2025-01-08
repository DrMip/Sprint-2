import numpy
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import get_points_from_data as gp
import pandas as pd
#import xlrd


def excel_reader():
    file_location_array = ['parabolic_workbook.xlsx']

    for i in range(len(file_location_array)):
        df = pd.read_excel(file_location_array[i])
        column_names = df.columns
        # print the column names
        # get the values for a given column
        values = df[column_names[i]].values

        # get a data frame with selected columns
        FORMAT = ['t', 'x', 'y', 'z']
        df_selected = df[FORMAT]
        res = []
        for column in df.columns:
            # Storing the rows of a column
            # into a temporary list
            li = df[column].tolist()

            # appending the temporary list
            res.append(li)

        # Printing the final list
        return res


# parabolic function
def parabolic(x, a, b, c):
    return a * x * x + b * x + c


def linear(y, a, b):
    return a * y + b


def pol_fourth_order(x, a, b, c, d, e):
    return a*x**4 + b*x**3 + c*x**2 + d*x + e


def parabolic_fit_2(array_of_time_and_locs, i):
    # this is for the altitude - z

    t_data = np.asarray(array_of_time_and_locs[0])
    y_data = np.asarray(array_of_time_and_locs[i])
    if i ==3:
        initial_guess = [-9.8 / 2, 0.0, 5486]

    else:
        initial_guess = [0.1, 0.1, 35]
    params, covariance = curve_fit(parabolic, t_data, y_data, p0=initial_guess)
    # Extract the parameters
    a, b, c = params

    # Create a range of x values for the curve change value of "20" to max number or data points i didnt know how to get max size of the data sheet
    t_fit = np.linspace(min(t_data), max(t_data), 20)

    # Calculate the y values for the fitted curve
    y_fit = parabolic(t_fit, a, b, c)
    '''
    plt.figure(figsize=(16, 12))
    plt.scatter(t_data, y_data, label="Data")
    plt.plot(t_fit, y_fit, label="Tanh Fit", color="red")
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.xlabel("x", fontsize=17)
    plt.ylabel("y", fontsize=17)
    plt.title("x vs time", fontsize=22)

    equation = f"y = {a:.2f} *t^2 {b:.2f} * t + {c:.2f})"
    #print("Equation:", equation)

    text_x = 8  # x-coordinate
    text_y = 16  # y-coordinate

    #plt.show()
    '''
    return a, b, c



def linear_fit_2(x_or_y, array):
    # this can be used for the longitude and the langitude

    t_data = np.asarray(array[0])
    if x_or_y == 'x':
        i = 1
    else:
        i = 2
    y_data = np.asarray(array[i])

    initial_guess = [10, 10]
    params, covariance = curve_fit(linear, t_data, y_data, p0=initial_guess)
    # Extract the parameters
    a, b = params

    # Create a range of x values for the curve change value of "20" to max number or data points i didnt know how to get max size of the data sheet
    x_fit = np.linspace(min(t_data), max(t_data), 150)

    # Calculate the y values for the fitted curve
    y_fit = linear(x_fit, a, b)
    '''
    plt.figure(figsize=(16, 12))
    plt.scatter(t_data, y_data, label="Data")
    plt.plot(x_fit, y_fit, label="Tanh Fit", color="red")
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.xlabel("x", fontsize=17)
    plt.ylabel("y", fontsize=17)
    plt.title("x vs time", fontsize=22)

    #equation = f"{x_or_y} = {a:.2f} *t + {b:.2f} "
    #print("Equation:", equation)

    text_t = 8  # x-coordinate
    text_y = 16  # y-coordinate

    #plt.show()
    '''
    return a, b


def parabolic_fit_2(array_of_time_and_locs, i):
    # this is for the altitude - z

    t_data = np.asarray(array_of_time_and_locs[0])
    y_data = np.asarray(array_of_time_and_locs[i])

    initial_guess = [1, 1, -9.8 / 2, 0.0, 5486]

    params, covariance = curve_fit(pol_fourth_order, t_data, y_data, p0=initial_guess)
    # Extract the parameters
    a, b, c, d, e = params

    # Create a range of x values for the curve change value of "20" to max number or data points i didnt know how to get max size of the data sheet
    t_fit = np.linspace(min(t_data), max(t_data), 20)

    # Calculate the y values for the fitted curve
    y_fit = pol_fourth_order(t_fit, a, b, c, d, e)
    '''
    plt.figure(figsize=(16, 12))
    plt.scatter(t_data, y_data, label="Data")
    plt.plot(t_fit, y_fit, label="Tanh Fit", color="red")
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.xlabel("x", fontsize=17)
    plt.ylabel("y", fontsize=17)
    plt.title("x vs time", fontsize=22)

    equation = f"y = {a:.2f} *t^2 {b:.2f} * t + {c:.2f})"
    #print("Equation:", equation)

    text_x = 8  # x-coordinate
    text_y = 16  # y-coordinate

    #plt.show()
    '''
    return a, b, c, d, e



def solve_x_y_when_t_is_zero(array):

    z_a_b_c = parabolic_fit_2(array, 3)

    both_ts = np.roots(z_a_b_c)
    t_ground = min(both_ts)

    #y_a_b = linear_fit_2('y', array)
    y_a_b_c = parabolic_fit_2(array, 1)
    y0 = y_a_b_c[0]*t_ground**2 +y_a_b_c[1]*t_ground + y_a_b_c[2]

    #x_a_b = linear_fit_2('x', array)
    x_a_b_c = parabolic_fit_2(array, 2)

    x0 = x_a_b_c[0]*t_ground**2 +x_a_b_c[1]*t_ground + x_a_b_c[2]

    return x0, y0

def turns_excel_data_into_array(index_of_rocket = 1):
    x = gp.get_data()[index_of_rocket]
    array = x.get_locations()

    data_array = []
    time = []

    latt = []

    long = []

    alt = []

    for i in array:
        time.append(i.get_time() - 1736300020)
        latt.append(i.get_latitude())
        long.append(i.get_longitude())
        alt.append(i.get_altitude())

    #x - lattitude
    #y - longitude
    data_array.append(time)
    data_array.append(latt)
    data_array.append(long)
    data_array.append(alt)
    return data_array


def calculate_landing_point(rocket_numer):
    array = turns_excel_data_into_array(rocket_numer)

    coords = solve_x_y_when_t_is_zero(array)

    return coords


def calculate_locs_for_all_rocket_landings():
    array_of_landing_locations = []
    for i in range(len(gp.get_data()) - 1):
        array_of_landing_locations.append(calculate_landing_point(i + 1))
    return array_of_landing_locations

#array = calculate_locs_for_all_rocket_landings()
#print(array)





































































'''

from scipy.optimize import minimize


# Function to calculate velocity from position using central differences
def calculate_velocity(positions, times):
    velocities = []
    dt = times[1] - times[0]  # Assume constant time intervals
    for i in range(1, len(positions) - 1):
        # Central difference method for velocity estimation
        vx = (positions[i + 1][0] - positions[i - 1][0]) / (2 * dt)
        vy = (positions[i + 1][1] - positions[i - 1][1]) / (2 * dt)
        vz = (positions[i + 1][2] - positions[i - 1][2]) / (2 * dt)
        velocities.append([vx, vy, vz])
    return np.array(velocities)


# Runge-Kutta step for 3D motion with air resistance
def runge_kutta_step_3d(t, state, dt, k, m, g=9.81):
    # state = [x, y, z, vx, vy, vz]
    x, y, z, vx, vy, vz = state
    v = np.sqrt(vx ** 2 + vy ** 2 + vz ** 2)

    # Forces from air resistance
    F_air_x = -k * v * vx
    F_air_y = -k * v * vy
    F_air_z = -k * v * vz

    # Accelerations
    ax = F_air_x / m
    ay = (F_air_y - m * g) / m
    az = F_air_z / m

    # Runge-Kutta 4th order method
    k1 = np.array([vx, vy, vz, ax, ay, az])
    k2 = np.array([vx + 0.5 * dt * k1[3], vy + 0.5 * dt * k1[4], vz + 0.5 * dt * k1[5],
                   ax + 0.5 * dt * k1[3], ay + 0.5 * dt * k1[4], az + 0.5 * dt * k1[5]])
    k3 = np.array([vx + 0.5 * dt * k2[3], vy + 0.5 * dt * k2[4], vz + 0.5 * dt * k2[5],
                   ax + 0.5 * dt * k2[3], ay + 0.5 * dt * k2[4], az + 0.5 * dt * k2[5]])
    k4 = np.array([vx + dt * k3[3], vy + dt * k3[4], vz + dt * k3[5],
                   ax + dt * k3[3], ay + dt * k3[4], az + dt * k3[5]])

    # New state after the time step
    new_state = np.array([x, y, z, vx, vy, vz]) + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    return new_state


# Simulate the projectile motion with air resistance
def projectile_motion_with_air_resistance_3d(v0, theta, phi, h0, k, m, times):
    # Initial velocity components from angles (similar to before)
    theta = np.radians(theta)
    phi = np.radians(phi)
    vx0 = v0 * np.cos(theta) * np.cos(phi)
    vy0 = v0 * np.cos(theta) * np.sin(phi)
    vz0 = v0 * np.sin(theta)

    state = [0, h0, 0, vx0, vy0, vz0]  # Initial state: [x, y, z, vx, vy, vz]
    dt = 0.01  # Time step
    x_values, y_values, z_values = [0], [h0], [0]

    for t in times[1:]:  # Start from the second time point
        state = runge_kutta_step_3d(t, state, dt, k, m)
        x_values.append(state[0])
        y_values.append(state[1])
        z_values.append(state[2])

    return np.array([x_values, y_values, z_values]).T


# Objective function to minimize the error between simulated and observed trajectory
def objective(params, times, observed_positions):
    v0, theta, phi, h0, k, m = params
    simulated_positions = projectile_motion_with_air_resistance_3d(v0, theta, phi, h0, k, m, times)

    # Calculate the error (sum of squared differences)
    error = np.sum((simulated_positions - observed_positions) ** 2)
    return error


# Example observed positions (replace this with your actual data)
times = np.array([0, 1, 2, 3, 4, 5])  # Example times (in seconds)
observed_positions = np.array(
    [[0, 0, 20], [5, 5, 18], [10, 10, 16], [15, 15, 14], [20, 20, 12], [25, 25, 10]])  # Example positions (x, y, z)

# Calculate velocities from the observed positions
velocities = calculate_velocity(observed_positions, times)

# Initial guesses for the parameters
initial_guess = [40, 30, 45, 20, 0.1, 1]  # [v0, theta, phi, h0, k, m]

# Use scipy's minimize function to fit the parameters
result = minimize(objective, initial_guess, args=(times, observed_positions))

# Output the optimized parameters
print("Optimized parameters:", result.x)

# Simulate the motion with the optimized parameters
v0_opt, theta_opt, phi_opt, h0_opt, k_opt, m_opt = result.x
simulated_positions = projectile_motion_with_air_resistance_3d(v0_opt, theta_opt, phi_opt, h0_opt, k_opt, m_opt, times)

# Plot the observed and simulated trajectories
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(observed_positions[:, 0], observed_positions[:, 1], observed_positions[:, 2], 'o', label='Observed')
ax.plot(simulated_positions[:, 0], simulated_positions[:, 1], simulated_positions[:, 2], label='Simulated',
        linestyle='--')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Projectile Motion with Air Resistance (3D)')
ax.legend()
plt.show()

'''



