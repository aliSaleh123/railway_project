import numpy as np
import pandas as pd

class Crack:
    def __init__(self, start, end, conditions=None):

        if start <= end:
            self.start = start
            self.end = end
        else:
            self.end = start
            self.start = end

        self.centroid = (start + end) / 2

        if conditions is None:
            self.conditions = {}
        else:
            self.conditions = conditions

        self.transitions_mgt = {}

    def add_condition(self, date, condition):
        self.conditions[date] = condition

    def sort_conditions(self):
        # sort conditions based on the keys, which are dates
        self.conditions = dict(sorted(self.conditions.items()))

    def create_transitions_mgt(self, tonnage_dict=None):

        # conditions_map = {
        #     0: 0,
        #     0.3: 1,
        #     0.8: 2,
        #     1.3: 3,
        #     1.8: 4,
        #     2.3: 5,
        #     'unknown': -1
        # }

        self.transitions_mgt = {}

        if (not tonnage_dict) | (self.conditions is None):
            return

        dates = list(self.conditions.keys())
        depths = list(self.conditions.values())
        tonnage_at_dates = []

        tonnage_dict_dates = list(tonnage_dict.keys())
        for date in dates:

            # if isinstance(date, pd.Timestamp):
            #     date = date.to_numpy()

            # get tonnage at this specific date
            closest_date = tonnage_dict_dates[np.abs(tonnage_dict_dates - date.to_numpy()).argmin()]

            tonnage_at_dates.append(tonnage_dict[closest_date])

        for i in range(1, len(dates)):
            # transition_key = (conditions_map[depths[i - 1]], conditions_map[depths[i]])
            transition_key = (depths[i - 1], depths[i])

            num_days = abs((dates[i - 1] - dates[i]).days)

            # if isinstance(date, pd.Timestamp):
            #     num_days = abs((dates[i - 1] - dates[i]).days)
            # else:
            #     num_days = (dates[i].astype('datetime64[D]')-dates[i - 1].astype('datetime64[D]')).astype('int64')

            self.transitions_mgt[transition_key] = num_days * tonnage_at_dates[i - 1] / 1e6
