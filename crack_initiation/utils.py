from scipy.optimize import curve_fit
import pandas as pd
import numpy as np


def create_var_per_day(raw_data, variable_name):
    # The function calculates the damage indices per one day

    # n is the length of each time period in days.
    # The input dataframe should contain the 'Datum' and the 'variable_name' columns.

    # Group the dataframe by every n days
    grouped = raw_data.groupby('days')

    # Calculate the sum of values for each group
    result = grouped[variable_name].sum()

    # Get the first date and day of each group
    days = grouped['days'].first()

    # Combine the result and dates into a single dataframe
    variable_per_time = pd.concat([days, result], axis=1)

    # Rename the columns
    variable_per_time.columns = ['days', variable_name]

    return variable_per_time




def func1(x, a):
    # 2nd order polynomial function to model the accumulated damage
    # c = 0 because we know that accumulated damage at day=0 is 0
    return 0 * np.array(x) + a


def extrapolate_df(data_frame, x_name, y_name, last_x=None):
    x_values = data_frame[x_name].values
    y_values = data_frame[y_name].values

    params, _ = curve_fit(func1, x_values, y_values)

    if last_x is None:
        last_x = int(data_frame['days'].max())

    x = get_missing_values(data_frame['days'].values, last_x)
    y = func1(x, *params)

    data_frame2 = pd.concat([data_frame, pd.DataFrame({x_name: x, y_name: y})])

    sorted_df = data_frame2.sort_values('days').reset_index(drop=True)

    return sorted_df


def get_missing_values(values, last_value):
    missing_values = []
    for i in range(last_value):
        if i not in values:
            missing_values.append(i)

    return missing_values
