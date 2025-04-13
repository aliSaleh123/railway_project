import pandas as pd
import re
import itertools
import dill


def get_spoors(spoor_details_col):
    # split by ";" all the column, then explode the column, keep unique values, and change to list
    spoor_details = spoor_details_col.apply(lambda x: x.split(';')).explode().unique().tolist()

    # Find all matches in the string (all characters between "sp_" and "_")
    matches = re.findall(r"sp_(.*?)_", "_".join(spoor_details))

    # split based on "-", "/", and "|"
    split_string = [re.split(r"[-/|]", match) for match in matches]

    # flatten the list
    spoors_list = list(itertools.chain(*split_string))

    # keep only unique values
    spoors_list = list(set(spoors_list))

    return spoors_list


def get_spoortak_names(
        raw_eddy_current_path,
        spoortak_dict_path
):
    measuremens_ec = pd.read_excel(raw_eddy_current_path, skiprows=[0])

    spoortak_dict = {}

    # group by spoortak and get spoor names in each spoortak
    grouped = measuremens_ec.groupby('spoortak')

    for spoortak, group in grouped:
        geo_code = spoortak.split('_')[0]
        if geo_code not in spoortak_dict.keys():
            spoortak_dict[geo_code] = []

        spoortak_dict[geo_code].append([spoortak, get_spoors(group['gebiedsindeling'])])

    with open(spoortak_dict_path, 'wb') as f:
        dill.dump(spoortak_dict, f)
