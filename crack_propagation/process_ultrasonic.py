import pandas as pd
import dill
import re
from .crack import Crack


def combine_dicts(dict1, dict2):
    for key, value in dict2.items():
        if key in dict1:
            dict1[key] = max(dict1[key], value)
        else:
            dict1[key] = value

    return dict1


class Section:
    def __init__(self, object_oms, side, spoor_names, Geo, cracks):
        self.object_oms = object_oms        # to specify crack
        self.side = side                    # to specify crack
        self.spoor_names = spoor_names      # to relate to the EC measurements
        self.Geo = Geo                      # specify the geocode of the section
        self.spoortak = ''                  # this is filled later based on the spoor_names

        self.cracks = cracks  # cracks based on crack_Cls

        self.tonnage = {}  # this is filled later

    def check_overlaps(self):

        # get all the vertices that represent the start and the end of each segment (crack)
        vertices = sorted(
            set([crack.start for crack in self.cracks if crack.start != crack.end] +
                [crack.end for crack in self.cracks if crack.start != crack.end]
                ))

        # create new cracks based on the extracted vertices
        new_cracks = [Crack(vertices[i], vertices[i + 1]) for i in range(len(vertices) - 1)]

        # add cracks that have equal start and end
        new_cracks += [crack for crack in self.cracks if crack.start == crack.end]

        # fill the new cracks' conditions
        for new_crack in new_cracks:
            for crack in self.cracks:
                if crack.start <= new_crack.centroid <= crack.end:
                    # add all the conditions of the first crack to that of the next crack
                    # if two conditions at the same date exist, keep the worst condition.
                    new_crack.conditions = combine_dicts(new_crack.conditions.copy(), crack.conditions.copy())

        # create masked cracks by keeping only the cracks that have conditions
        self.cracks = [crack for crack in new_cracks if len(crack.conditions) > 0]

        # sort the conditions of the kept cracks based on the dates
        for crack in self.cracks:
            crack.sort_conditions()

        # sort the cracks based on crack.start
        self.cracks = sorted(self.cracks, key=lambda x: x.start)

    def create_tonnage_dict(self, tonnage_dfs):
        self.tonnage = {}
        for tonnage_df in tonnage_dfs:
            row = tonnage_df[tonnage_df['Spoortak_Identificatie'] == self.spoortak]
            if len(row) >= 1:
                self.tonnage[row['Periode_van'].values[0]] = row['Dagtonnage_totaal'].values[0]

    def create_cracks_transitions(self):
        for crack in self.cracks:
            crack.create_transitions_mgt(self.tonnage)


def process_section_US(specific_section_cracks, object_oms, side):
    # the input is group of cracks that has the same geocode, ObjectOms, and side
    # create different cracks based on the positions

    # group according to position
    grouped_by_location = specific_section_cracks.groupby(['KilometerVan', 'KilometerTot'])

    cracks = []
    for name, specific_location in grouped_by_location:
        # sort based on date
        specific_location = specific_location.sort_values(by='Datum')

        # create the conditions dictionary
        conditions = dict(zip(specific_location['Datum'].tolist(), specific_location['crackDepth'].tolist()))

        # create a crack within the section
        cracks.append(Crack(name[0], name[1], conditions))

    # keep unique spoor names of each column
    spoor_names = specific_section_cracks.explode('spoorNames')['spoorNames'].unique().tolist()
    geo_code = specific_section_cracks.explode('Geocode')['Geocode'].unique()[0]

    return Section(object_oms, side, spoor_names, geo_code, cracks)


def get_spoortak(spoortaks_list, spoor_names_section):
    spoortaks = []
    for [spoortak, spoor_names] in spoortaks_list:

        for spoor_name_sec in spoor_names_section:
            if spoor_name_sec in spoor_names:
                spoortaks.append(spoortak)
                break

    return spoortaks


def process_us_data(raw_ultrasonic_path, processed_ultrsonic_path, spoortak_dict_path, tonnage_dfs_path):
    # load spoortak dict
    with open(spoortak_dict_path, 'rb') as f:
        spoortak_dict = dill.load(f)

    with open(tonnage_dfs_path, 'rb') as f:
        tonnage_dfs = dill.load(f)

    # read the US measurements
    # noinspection PyTypeChecker
    measurements_df = pd.read_excel(
        raw_ultrasonic_path,
        dtype={'Geocode': str},
        sheet_name='Sheet1',
        usecols=['Datum', 'KilometerTot', 'KilometerVan', 'US_Scheurdiepte',
                 'ObjectOms', 'Spoor US', 'US_Scheurdiepte', 'Been', 'Geocode']
    )

    # delete rows that contain nan in the following columns
    measurements_df = measurements_df.dropna(subset=['KilometerTot', 'KilometerVan', 'US_Scheurdiepte'])

    # Convert NaN values to empty string
    measurements_df = measurements_df.fillna('')

    # convert the 'ObjectOms' column to lowercase
    measurements_df['ObjectOms'] = measurements_df['ObjectOms'].str.lower()
    measurements_df['Spoor US'] = measurements_df['Spoor US'].str.lower()

    measurements_df['crackDepth'] = measurements_df['US_Scheurdiepte']

    # -- - -- - -- - -- - -- - -- - -- - -- - -- - -- - -- - -- - -- - -- - -- - --#
    # convert to L R and any for non left or right sides
    def get_spoors(x):
        if x in ['L', 'R']:
            return x
        else:
            return 'any'

    measurements_df['Been'] = measurements_df['Been'].apply(get_spoors)
    # -- - -- - -- - -- - -- - -- - -- - -- - -- - -- - -- - -- - -- - -- - -- - --#
    # get the spoor names based on 'Spoor US' and 'ObjectOms'
    # the pattern for getting the spoor names list of each row by splitting using -, /, or |
    pattern = r"[-/|]"

    def get_spoors(object_oms):
        if 'spoor' in object_oms:
            return re.split(pattern, object_oms.split("spoor ")[1])
        else:
            return []

    def keep_unique(lst):
        return list(set(lst))

    # create spoorNames based on 'Spoor US' and 'ObjectOms'
    measurements_df['spoorNames'] = (
            measurements_df['Spoor US'].apply(lambda x: re.split(pattern, x)) + measurements_df['ObjectOms'].apply(
        get_spoors)
    ).apply(keep_unique)

    # -- - -- - -- - -- - -- - -- - -- - -- - -- - -- - -- - -- - -- - -- - -- - --#
    # create empty processed sections
    processed_sections = {key: [] for key in measurements_df['Geocode'].values}

    # group based on geocode
    grouped_bygeocode = measurements_df.groupby('Geocode')

    for geo_code, group in grouped_bygeocode:

        # group the data based on spoortak and link/rechts
        grouped_bysection = group.groupby(['ObjectOms', 'Been'])

        for section_name, specific_section_cracks in grouped_bysection:
            # process the section and append it to the corresponding geocode
            processed_sections[geo_code].append(
                process_section_US(specific_section_cracks, section_name[0], section_name[1])
            )

    #  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  #
    # check overlapping regions for each section
    for geo_code in processed_sections:
        for section in processed_sections[geo_code]:
            section.check_overlaps()

    #  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  #
    # get the spoortak of each section
    for geo_code in processed_sections:
        for sectionUS in processed_sections[geo_code]:

            if geo_code in spoortak_dict.keys():
                spoortaks = get_spoortak(spoortak_dict[geo_code], sectionUS.spoor_names)
                if len(spoortaks) > 0:
                    sectionUS.spoortak = spoortaks[0]
                else:
                    sectionUS.spoortak = 'unavailable'
            else:
                sectionUS.spoortak = 'unavailable'

    for geo_code in processed_sections:
        for section in processed_sections[geo_code]:
            section.create_tonnage_dict(tonnage_dfs)

    # create cracks transitions for each crack within each section
    for geo_code in processed_sections:
        for section in processed_sections[geo_code]:
            section.create_cracks_transitions()

    # for geo in processed_sections:
    #     for sec in processed_sections[geo]:
    #         sec.tonnage = {}
    #         for i in range(4):
    #             row = tonnageList[i][tonnageList[i]['Spoortak_Identificatie'] == sec.spoortak]
    #             if len(row) == 1:
    #                 sec.tonnage[row['Periode_van'].values[0]] = row['Dagtonnage_totaal'].values[0]

    #  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  #
    # save the results
    with open(processed_ultrsonic_path, 'wb') as f:
        dill.dump(processed_sections, f)
