# -*- coding: utf-8 -*-
"""

@author: Ali Mohamad Saleh

Reads Excel sheets and Pickles them to make reading them easier in futrue
"""

import os
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
import pickle


def pickle_all_files(files_dir, parallel=False):
    def pickle_file(file_name):
        df = pd.read_excel(os.path.join(files_dir, file_name), nrows = 100)

        new_file_name = os.path.splitext(file_name)[0] + ".pkl"

        print(new_file_name)

        with open(os.path.join(files_dir, f"_pickled_{new_file_name}"), 'wb') as f:
            pickle.dump(df, f)

    # get all the names of the files to read them
    files = [f for f in os.listdir(files_dir) if os.path.isfile(os.path.join(files_dir, f))]

    if parallel:
        with ThreadPoolExecutor() as executor:
            executor.map(pickle_file, files)
    else:
        for file in files:
            pickle_file(file)


def combine_pickled_raw_files(files_dir, combined_raw_data_path):
    # get all pickled files
    files = [f for f in os.listdir(files_dir) if f.startswith('_pickled_')]

    # read all pickled data frames
    dfs = []
    for file in files:
        with open(os.path.join(files_dir, file), 'rb') as f:
            dfs.append(pickle.load(f))

    # combine all pickled data frames
    data_frame = pd.concat(dfs).reset_index(drop=True)

    # sort the dataframe based on the date column
    data_frame = data_frame.sort_values('Datum').reset_index(drop=True)

    # Pickle the combined raw data
    with open(combined_raw_data_path, 'wb') as f:
        pickle.dump(data_frame, f)


def remove_pickled_files(files_dir):
    # get all pickled files
    files = [f for f in os.listdir(files_dir) if f.startswith('_pickled_')]

    for file in files:
        os.remove(os.path.join(files_dir, file))


def combine_raw_data(raw_data_dir, combined_raw_data_path):
    pickle_all_files(raw_data_dir, parallel=True)

    combine_pickled_raw_files(raw_data_dir, combined_raw_data_path)

    remove_pickled_files(raw_data_dir)



def test():
    from config import RAW_DATA_DIR, PROCESSED_DATA_DIR

    combine_raw_data(
        raw_data_dir=RAW_DATA_DIR / "usage",
        combined_raw_data_path=PROCESSED_DATA_DIR / "usage" / "combined_raw_data.pkl"
    )



# if __name__ == '__main__':
#     from pathlib import Path
#     import os
#
#     main_dir = Path(__file__).parent.parent.resolve()
#
#     combine_raw_data(
#         raw_data_dir=os.path.join(main_dir, "data", "raw", "usage"),
#         combined_raw_data_path=os.path.join(main_dir, "data", "processed", "usage", "combined_raw_data.pkl")
#     )
