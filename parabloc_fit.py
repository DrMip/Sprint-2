import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

import pandas as pd
import xlrd


def excel_reader():
    file_location_array = ['parabolic_workbook.xlsx']


    for i in range(len(file_location_array)):
        df = pd.read_excel(file_location_array[i])

        column_names = df.columns
        #print the column names
        #get the values for a given column
        values = df[column_names[i]].values

        #get a data frame with selected columns
        FORMAT = ['x', 'y']
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


#parabolic function
def parabolic(x, a, b, c):
    return a*x*x + b*x + c


def parabolic_fit_2():

    array_x_y_loc = excel_reader()

    x_data = np.asarray(array_x_y_loc[0])
    y_data = np.asarray(array_x_y_loc[1])

    initial_guess = [-0.15, 0.0, 0.0]
    params, covariance = curve_fit(parabolic, x_data, y_data, p0=initial_guess)
    # Extract the parameters
    a, b, c = params

    # Create a range of x values for the curve change value of "127" to max number or data points i didnt know how to get max size of the data sheet
    x_fit = np.linspace(min(x_data), max(x_data), 137)
    # Calculate the y values for the fitted curve
    y_fit = parabolic(x_fit, a, b, c)

    plt.figure(figsize=(16, 12))
    plt.scatter(x_data, y_data, label="Data")
    plt.plot(x_fit, y_fit, label="Tanh Fit", color="red")
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.xlabel("x", fontsize=17)
    plt.ylabel("y", fontsize=17)
    plt.title("x vs time", fontsize=22)

    equation = f"y = {a:.2f} *x^2 {b:.2f} * x + {c:.2f})"
    print("Equation:", equation)

    text_x = 8  # x-coordinate
    text_y = 16  # y-coordinate

    plt.show()



parabolic_fit_2()


