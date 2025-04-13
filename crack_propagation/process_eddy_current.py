import pandas as pd
import dill
import numpy as np
import itertools
import re
from .crack import Crack
from .config import DEPTH_MAP_EC

def get_section_range(grouped_dates):
    # grouped_dates is the section grouped by Dates
    # return possible range for the section that has values for all dates

    from_values = np.zeros(len(grouped_dates))
    to_values = np.zeros(len(grouped_dates))

    i = 0
    for date_name, group in grouped_dates:
        from_values[i] = group['from'].min()
        to_values[i] = group['to'].max()
        i += 1

    return [max(from_values), min(to_values)]


def get_condition(df_in, position):
    # input is:
    # df_in: dataframe for one date measurements covering the section
    # the df_in should contain "from", "to", and "condition" columns
    # position: position within the section
    condition = []
    for i, row in df_in.iterrows():
        if (row['from'] < position) & (position < row['to']):
            # update the condition
            condition.append(row['condition'])
        else:
            # we already found an intervals and this interval does not contain the point
            # which means that no more intervals can contain the point, so return what was already found
            if len(condition) > 0:
                return max(condition)

    if len(condition) > 0:
        return max(condition)

    # if point is outside the range of values return 'unknown'
    if (position < df_in['from'].min()) | (position > df_in['to'].max()):
        return 'unknown'

    # point is in none of the intervals
    # return the condition of the interval closest to the point

    # find the row with the closest "from" value to point P

    # Apply the lambda function to each row of the DataFrame and assign the result to a new column called "distance"
    df_in['distance'] = df_in.apply(
        lambda row: min(abs(position - row['from']), abs(position - row['to'])),
        axis=1
    )

    # Find the index of the row with the minimum distance
    min_dist_index = df_in['distance'].idxmin()

    # Select the row with the minimum distance
    return df_in.loc[min_dist_index]['condition']


def process_DF_specific_date(specific_date, range_section, only_return_vertices=False):
    # this function is used to get the important vertices within the section at specific date
    # these vertices represent the position where the condition changes

    # discretize based on ranges
    vertices = np.concatenate((specific_date['from'].values, specific_date['to'].values))

    # Keep only unique and sorted values
    vertices = np.unique(vertices)

    #    #delete vertices that are out of the range
    #    vertices = vertices[(vertices >= range_section[0]) & (vertices <= range_section[1])]

    # get centroids based on the values of these vertices
    centroids = np.zeros(len(vertices) - 1)
    for i in range(len(vertices) - 1):
        centroids[i] = (vertices[i] + vertices[i + 1]) / 2

    # get the condition at each cetroid
    conditions = np.zeros(len(centroids))
    for i in range(len(centroids)):
        conditions[i] = get_condition(specific_date.copy(), centroids[i])

    # combine the consecutive segments that have the same conditions
    toDelete = []
    for i in range(len(conditions) - 1):
        if conditions[i] == conditions[i + 1]:
            toDelete.append(i + 1)

    # delete the unnecessary vertices
    conditions = [val for i, val in enumerate(conditions) if i not in toDelete]
    vertices = [val for i, val in enumerate(vertices) if i not in toDelete]

    if only_return_vertices:
        return vertices

    # no need for date, spoortak, links because they will be representing the group
    # keep only from, to, condition
    specific_date = pd.DataFrame({
        'from': vertices[:-1],
        'to': vertices[1:],
        'condition': conditions
    })

    return specific_date


class Section():
    def __init__(self, spoortak, side, spoor_names, cracks):
        self.spoortak = spoortak
        self.side = side
        self.cracks = cracks

        self.geo = self.spoortak.split("_")[0]

        self.spoor = spoor_names

        self.tonnage = {}

    def check_geo(self, geo):
        if geo == self.geo:
            return True
        else:
            return False

    def check_side(self, side):

        if side not in ['L', 'R']:
            # side can't be checked or it is true
            return True

        if side == self.side:
            return True
        else:
            return False

    def check_spoor(self, spoor):
        if (self.spoor in spoor[0]) | (self.spoor in spoor[1]):
            return True
        else:
            return False

    def is_in_section(self, geo, spoor, side):
        if self.check_geo(geo) & self.check_side(spoor) & self.check_spoor(side):
            return True
        else:
            return False

    def create_tonnage_dict(self, tonnage_dfs):
        self.tonnage = {}

        for tonnage_df in tonnage_dfs:
            row = tonnage_df[tonnage_df['Spoortak_Identificatie'] == self.spoortak]
            if len(row) >= 1:
                self.tonnage[row['Periode_van'].values[0]] = row['Dagtonnage_totaal'].values[0]

    def create_cracks_transitions(self):
        for crack in self.cracks:
            crack.create_transitions_mgt(self.tonnage)



def process_section(specific_section, spoortak, side):
    # group by dates
    grouped_dates = specific_section.groupby('datum')

    # range of the section where measurements exist for all the dates
    range_section = get_section_range(grouped_dates)

    vertices = []
    for date_name, group in grouped_dates:
        # each group has specific 'spoortak', 'links/r' and 'datum'
        vertices.append(process_DF_specific_date(group.copy(), range_section, only_return_vertices=True))

    # combine the vertices in one array, delete the repeated values and sort the values
    vertices = np.unique(np.concatenate(vertices))

    # get spoor names
    spoor_names = get_spoors(specific_section['gebiedsindeling'])

    #    sectionCracks = [ Crack(vertices[i], vertices[i+1]) for i in range(len(vertices)-1)]

    # initialize the processed section
    processed_section = Section(
        spoortak, side, spoor_names, [Crack(vertices[i], vertices[i + 1]) for i in range(len(vertices) - 1)])

    # fill the processed section with the crack history
    for date_name, group in grouped_dates:
        for crack in processed_section.cracks:
            # get condition of this crack position
            condition = get_condition(group.copy(), crack.centroid)

            # add the condition with the corresponding date
            crack.add_condition(date_name, condition)

    return processed_section


def get_spoors(spoor_details_column):
    # split by ";" all the column, then explode the column, keep unique values, and change to list
    spoor_details = spoor_details_column.apply(lambda x: x.split(';')).explode().unique().tolist()

    # Find all matches in the string (all characters between "sp_" and "_")
    matches = re.findall(r"sp_(.*?)_", "_".join(spoor_details))

    # split based on "-", "/", and "|"
    split_string = [re.split(r"[-/|]", match) for match in matches]

    # flatten the list
    spoors_list = list(itertools.chain(*split_string))

    # keep only unique values
    spoors_list = list(set(spoors_list))

    return spoors_list


def process_ec_data(raw_eddy_current_path, tonnage_dfs_path, processed_eddy_current_path):
    with open(tonnage_dfs_path, 'rb') as f:
        tonnage_dfs = dill.load(f)

    # create a dictionary that convert the specifications of the depth range to an average depth
    # depth_dict = {
    #     'Geen gebrek': 0,
    #     '0,1 – 0,5 mm': 0.3,
    #     '0,6 – 1,0 mm': 0.8,
    #     '1,1 – 1,5 mm': 1.3,
    #     '1,6 – 2,0 mm': 1.8,
    #     'Diepte > 2,1 mm': 2.3
    # }

    measuremens_EC_df = pd.read_excel(
        raw_eddy_current_path,
        skiprows=[0],
        engine='xlrd'
    )

    # remove the hours, minutes, seconds, and microseconds from the date
    measuremens_EC_df['datum'] = measuremens_EC_df['datum'].apply(
        lambda ts: ts.replace(hour=0, minute=0, second=0, microsecond=0))

    # create the condition column
    # use the map() method to create a new column based on the depth dictionary
    measuremens_EC_df['condition'] = measuremens_EC_df['diepteklasse'].map(DEPTH_MAP_EC)

    # get the geocode
    measuremens_EC_df['Geo'] = measuremens_EC_df['spoortak'].str.split('_').str[0]

    # get the location km from and to
    measuremens_EC_df['km tot'] = measuremens_EC_df['km tot'].str.split('#').str[0]
    measuremens_EC_df['km van'] = measuremens_EC_df['km van'].str.split('#').str[0]

    # change location from string to number
    measuremens_EC_df['km tot'] = measuremens_EC_df['km tot'].str.replace(',', '').astype(float)
    measuremens_EC_df['km van'] = measuremens_EC_df['km van'].str.replace(',', '').astype(float)

    # from is the minimum between "km van" and "km tot" because they are not always sorted
    measuremens_EC_df['from'] = measuremens_EC_df[['km van', 'km tot']].min(axis=1)
    measuremens_EC_df['to'] = measuremens_EC_df[['km van', 'km tot']].max(axis=1)

    # modify 'to' column where 'from' is equal to 'to'
    mask = measuremens_EC_df['from'] == measuremens_EC_df['to']
    measuremens_EC_df.loc[mask, 'to'] += measuremens_EC_df.loc[mask, 'lengte']

    # delete unnecessary columns
    measuremens_EC_df = measuremens_EC_df.drop(['km van', 'km tot', 'lengte', 'squat', 'diepteklasse'], axis=1)

    # create a dict for all the sections
    processed_sections = {key: [] for key in measuremens_EC_df['Geo'].values}

    # group based on geocode
    measuremens_EC_df_grouped = measuremens_EC_df.groupby('Geo')

    for geo_code, group in measuremens_EC_df_grouped:

        # group the data based on spoortak and link/rechts
        grouped_new = group.groupby(['spoortak', 'links/r'])

        for section_name, specific_section in grouped_new:
            # process the section and append it to the corresponding geocode
            processed_sections[geo_code].append(process_section(specific_section, section_name[0], section_name[1][0]))

    # create tonnage dict for every section
    for geo_code in processed_sections:
        for section in processed_sections[geo_code]:
            section.create_tonnage_dict(tonnage_dfs)

    # create cracks transitions for each crack within each section
    for geo_code in processed_sections:
        for section in processed_sections[geo_code]:
            section.create_cracks_transitions()

    with open(processed_eddy_current_path, 'wb') as f:
        dill.dump(processed_sections, f)



