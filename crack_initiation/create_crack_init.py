import dill
import numpy as np
from .config import di_naming_pattern, num_pattern
from sklearn.neighbors import KernelDensity


def get_crack_init_times(di_df):
    """
    return:
        crack_init_time:
            the number of days required to start crack initiation
    """
    days = np.array(di_df["days"])
    acc_di = np.array(di_df["accumulated_damage_Index"])
    crack_init_time = []

    for current_index, di in enumerate(acc_di):
        # the event occur after accumulation of 1
        di_event = di + 1

        if di_event > acc_di[-1]:
            last_index = current_index
            break

        # get the index of the event
        event_index = np.argmin(abs(di_event - acc_di))

        # get the day of this index
        current_day = days[current_index]
        day = days[event_index]

        crack_init_time.append(day - current_day)

    # plt.plot(days[:last_index], crack_init_time)

    return crack_init_time


def create_crack_init_times(di_dfs_dir, crack_init_times_path):
    # Get all files of the di_df results
    result_files = [file.name for file in di_dfs_dir.iterdir() if di_naming_pattern["read"].match(file.name)]

    # get the number of days to reach crack under for different rail radii
    crack_init_times = {}

    for filename in result_files:

        # get the variables values from the file name
        numbers = num_pattern.findall(filename)
        extracted_numbers = [float(num[0]) if '.' in num[0] else int(num[0]) for num in numbers]

        # h, radius = extracted_numbers
        h, radius, cant, coeff = extracted_numbers

        # cluster the data based on radius and cant
        key = (radius, cant)

        if key not in crack_init_times:
            crack_init_times[key] = []

        # read the df
        with open(di_dfs_dir / filename, 'rb') as f:
            di_df = dill.load(f)

        crack_init_times[key] += get_crack_init_times(di_df)

    with open(crack_init_times_path, 'wb') as f:
        dill.dump(crack_init_times, f)


def create_crack_init_dists(crack_init_times_path, crack_init_dists_path, data_factor=1):
    """

    Args:
        crack_init_times_path: path of times inputs
        crack_init_dists_path: path of distributions resulted
        data_factor: to convert data from days to other time units

    """
    with open(crack_init_times_path, 'rb') as f:
        crack_init_times = dill.load(f)

    # get the distributions using the times
    crack_init_dists = get_crack_init_dists(crack_init_times, data_factor)

    with open(crack_init_dists_path, 'wb') as f:
        dill.dump(crack_init_dists, f)


def get_crack_init_dists(crack_init_times, data_factor=1):
    """

    Args:
        crack_init_times (dict): dictionary of times to reach crack initiation at different conditions
        data_factor: to convert data from days to other time units

    Returns:
        crack_init_dists
    """

    crack_init_dists = {}
    for key in crack_init_times:
        data = np.array(crack_init_times[key]) * data_factor

        # Reshape data for KDE (requires 2D input)
        data = data[:, np.newaxis]

        crack_init_dists[key] = KernelDensity(kernel='gaussian', bandwidth=5 * data_factor).fit(data)

    return crack_init_dists
