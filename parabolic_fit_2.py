import numpy
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import get_points_from_data as gp
import pandas as pd

# import xlrd

'''
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
'''


# parabolic function
def parabolic(x, a, b, c):
    return a * x * x + b * x + c


def fourth_order(x, a, b, c, d, e):
    return a * x * x * x * x + b * x * x * x + c * x * x + d * x + e


def linear(y, a, b):
    return a * y + b


def parabolic_fit_2(array_of_time_and_locs, i):
    # this is for the altitude - z

    t_data = np.asarray(array_of_time_and_locs[0])
    y_data = np.asarray(array_of_time_and_locs[i])
    if i == 3:
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

    return a, b, c


def fourth_order_part_two(array_of_time_and_locs, i):
    # this is for the altitude - z
    t_data = np.asarray(array_of_time_and_locs[0])
    y_data = np.asarray(array_of_time_and_locs[i])

    if i == 3:
        initial_guess = [0.1, 0.1, -9.8 / 2, 0.0, 5486]
    else:
        initial_guess = [0.1, 0.1, 0.1, 0.1, 35]

    params, covariance = curve_fit(fourth_order, t_data, y_data, p0=initial_guess)
    # Extract the parameters
    a, b, c, d, e = params

    # Create a range of x values for the curve change value of "20" to max number or data points i didnt know how to get max size of the data sheet
    t_fit = np.linspace(min(t_data), max(t_data), 20)

    # Calculate the y values for the fitted curve
    y_fit = fourth_order(t_fit, a, b, c, d, e)

    plt.figure(figsize=(16, 12))
    plt.scatter(t_data, y_data, label="Data")
    plt.plot(t_fit, y_fit, label="Tanh Fit", color="red")
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.xlabel("x", fontsize=17)
    plt.ylabel("y", fontsize=17)
    plt.title("x vs time", fontsize=22)

    equation = f"y = {a:.2f} *t^2 {b:.2f} * t + {c:.2f})"
    # print("Equation:", equation)

    text_x = 8  # x-coordinate
    text_y = 16  # y-coordinate

    # plt.show()
    return a, b, c, d, e


'''
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

    return a, b
'''


def solve_x_y_when_t_is_zero(array):
    # z_a_b_c = parabolic_fit_2(array, 3)
    z_a_b_c = parabolic_fit_2(array, 3)
    both_ts = np.roots(z_a_b_c)
    t_ground = min(both_ts)

    # y_a_b = linear_fit_2('y', array)
    y_a_b_c = parabolic_fit_2(array, 1)

    y0 = y_a_b_c[0] * t_ground ** 2 + y_a_b_c[1] * t_ground + y_a_b_c[2]

    # x_a_b = linear_fit_2('x', array)
    x_a_b_c = parabolic_fit_2(array, 2)

    x0 = x_a_b_c[0] * t_ground ** 2 + x_a_b_c[1] * t_ground + x_a_b_c[2]

    return y0, x0


def turns_excel_data_into_array(index_of_rocket=1):
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

    # x - lattitude
    # y - longitude
    data_array.append(time)
    data_array.append(latt)
    data_array.append(long)
    data_array.append(alt)
    return data_array


def calculate_landing_point(rocket_numer: object) -> object:
    array = turns_excel_data_into_array(rocket_numer)

    coords = solve_x_y_when_t_is_zero(array)

    return coords


# print(calculate_landing_point(2))


def calculate_locs_for_all_rocket_landings():
    array_of_landing_locations = []
    for i in range(len(gp.get_data()) - 1):
        array_of_landing_locations.append(calculate_landing_point(i + 1))
    return array_of_landing_locations


def main():
    array = calculate_locs_for_all_rocket_landings()
    f = open("target_landing_parabolic_fit.txt", 'w')
    lst = []
    for tup in array:
        lst.append(tup)
    df = pd.DataFrame(lst)  # Skip the first tuple for headers

    # Write the DataFrame to an Excel file
    df.to_excel('output.xlsx', index=False)

    print("Data added successfully!")


main()
