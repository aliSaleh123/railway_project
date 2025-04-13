# -*- coding: utf-8 -*-
"""

@author: Ali Mohamad Saleh

Reads Excel sheets and Pickles them to make reading them easier in futrue
"""

import os
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
import pickle
from pathlib import Path


def read_all_files(data_dir, parallel=False):
    # get all the names of the files to read them
    files = [f for f in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, f))]

    def read_excel_file(file):
        df = pd.read_excel(os.path.join(data_dir, file))
        print(file)
        return df

    if parallel:
        with ThreadPoolExecutor() as executor:
            data_frames = list(executor.map(read_excel_file, files))
    else:
        data_frames = list(map(read_excel_file, files))

    return data_frames


def combine_raw_data_file(data_dir, results_file_path):
    # read all data files
    raw_data_frames = read_all_files(data_dir, parallel=False)

    # combine dataFrames, and reset the indices
    data_frame = pd.concat(raw_data_frames).reset_index(drop=True)

    # sort the dataframe based on the date column
    data_frame = data_frame.sort_values('Datum').reset_index(drop=True)

    # Pickle the raw data
    with open(results_file_path, 'wb') as f:
        pickle.dump(data_frame, f)


if __name__ == '__main__':
    main_dir = Path(__file__).parent.parent.resolve()

    raw_data_dir = os.path.join(main_dir, "data", "raw", "usage")
    raw_data_combined_path = os.path.join(main_dir, "data", "processed", "usage", 'combined_raw_data.pkl')

    combine_raw_data_file(raw_data_dir, raw_data_combined_path)
