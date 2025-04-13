# -*- coding: utf-8 -*-
"""

@author: Ali Mohamad Saleh

"""
import pickle
from .utils import extrapolate_df
from .utils import create_var_per_day


def create_mgt_df(combined_raw_data_path, mgt_data_path):
    # Read the pickled processed data
    with open(combined_raw_data_path, 'rb') as f:
        raw_df = pickle.load(f)

    # Calculate days in the raw dataframe
    raw_df['days'] = (raw_df['Datum'] - raw_df.loc[0, 'Datum']).dt.days

    # Create tonnage versus time
    mgt_df = create_var_per_day(raw_df, variable_name='Askwaliteit_aslast_ton')

    mgt_df = extrapolate_df(
        mgt_df,
        x_name='days',
        y_name='Askwaliteit_aslast_ton',
        last_x=None
    )

    # transform to million tons
    mgt_df["MGT"] = mgt_df['Askwaliteit_aslast_ton'] / 1e6

    # calculate accumulated MGT
    mgt_df['accumulated_MGT'] = mgt_df['MGT'].cumsum()

    # save the new data frame
    with open(mgt_data_path, 'wb') as f:
        pickle.dump(mgt_df, f)


def test():
    from config import PROCESSED_DATA_DIR, RESULTS_DIR

    create_mgt_df(
        combined_raw_data_path=PROCESSED_DATA_DIR / "usage" / "combined_raw_data.pkl",
        mgt_data_path=RESULTS_DIR / "mgt_df.pkl"
    )

