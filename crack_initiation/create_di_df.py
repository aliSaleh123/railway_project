# -*- coding: utf-8 -*-
"""

@author: Ali Mohamad Saleh

"""
import dill
import pandas as pd
import copy
from .config import input_parameters, di_naming_pattern
from .utils import extrapolate_df
from .utils import create_var_per_day


def wear_number_to_di(wear_number):
    # It returns the RCF damage index
    if wear_number < 20:
        return 0
    elif wear_number < 65:
        return 2.22e-7 * wear_number - 4.44e-6
    elif wear_number < 175:
        return -9.09e-8 * wear_number + 1.59e-5
    else:
        return 0


class CrackInitModel:
    # From "A hybrid predictive methodology for head checks in railway infrastructure"
    def __init__(self, crack_init_coefficients_path, h=1.3, radius=1500, cant=100, cof=0.3):
        """
        crack_init_coefficients_path:
            path to the crack initiation model coefficients
        h:
            rail profile, wear depth
        radius:
            radius of the track
        cant:
            can change because the train can settle down (between 0 and 160 mm)
        friction coefficient:
            only change per weather (time)
            Dry rail (steel on steel): 0.15–0.35
            Wet rail (e.g., due to rain): 0.05–0.15
            Lubricated rail (to reduce wear and noise): 0.01–0.05
        """

        # Read each sheet into a DataFrame
        self.coefficientsS1002 = pd.read_excel(crack_init_coefficients_path, sheet_name="S1002").values.reshape(-1)
        self.coefficientsHIT = pd.read_excel(crack_init_coefficients_path, sheet_name="HIT").values.reshape(-1)

        # these are the first four variables of the fitted models
        # these variables can be included also as variables instead of constants
        self.h = h  # rail profile, wear depth
        self.radius = radius  # radius of the track
        self.cant = cant  # can change because the train can settle down
        self.cof = cof  # only change per time

    # predict the wear number based on certain features
    def get_wear_number(self, X, wheel_type):
        # All the inputs to the regression model
        X = [self.h, self.radius, self.cant, self.cof] + X

        # the features of the regression model as a list
        X = [1, X[0], X[1], X[2], X[3], X[4], X[5], X[6], X[7],
             X[0] * X[1], X[0] * X[2], X[0] * X[3], X[0] * X[4], X[0] * X[5], X[0] * X[6], X[0] * X[7], X[1] * X[2],
             X[1] * X[3], X[1] * X[4], X[1] * X[5], X[1] * X[6], X[1] * X[7], X[2] * X[3], X[2] * X[4], X[2] * X[5],
             X[2] * X[6],
             X[2] * X[7], X[3] * X[4], X[3] * X[5], X[3] * X[6], X[3] * X[7], X[4] * X[5], X[4] * X[6], X[4] * X[7],
             X[5] * X[6],
             X[5] * X[7], X[6] * X[7], X[0] ** 2, X[1] ** 2, X[2] ** 2, X[3] ** 2, X[4] ** 2, X[5] ** 2, X[6] ** 2,
             X[7] ** 2]

        if wheel_type == 'HIT':
            # return element wise multiplication of coefficients and features
            return sum([x * cof for x, cof in zip(X, self.coefficientsHIT)])
        elif wheel_type == 's1002':
            # return element wise multiplication of coefficients and features
            return sum([x * cof for x, cof in zip(X, self.coefficientsS1002)])
        else:
            print('wheel type does not exist')

    # Calculate the damage indices for the following data set
    def get_damage_indices(self, data_frame):
        # Calculate the damage indices for the following data set

        # change the needed columns to list of lists
        input_data_list = data_frame[['Trein_snelheid_uit', 'Askwaliteit_aslast_kg', 'Longitudinal_stiffness',
                                      'Lateral_stiffness']].values.tolist()

        # get the wear number for each passed train
        wear_numbers = list(map(self.get_wear_number, input_data_list, data_frame['Wheel_type']))

        # get the RCF damage index for each passed train: damageIndex
        rcf_damage_indices = list(map(wear_number_to_di, wear_numbers))

        return rcf_damage_indices


def create_materials_conversion_dict(materials_properties_df):
    conversions_dict = {}
    for index, row in materials_properties_df.iterrows():
        conversions_dict[row['Voertuig_materieel_type']] = {
            'Wheel_type': row['Wheel_type'],
            'Longitudinal_stiffness': row['Longitudinal_stiffness'],
            'Lateral_stiffness': row['Lateral_stiffness']
        }
    return conversions_dict


def create_di_inputs_df(data_frame, materials_properties_df):
    di_inputs_df = copy.deepcopy(data_frame)

    # delete un-needed columns
    print('delete unnecessary columns')
    di_inputs_df = di_inputs_df.drop(
        ['Tijdstip', 'Meetunit', 'LocatieNaam', 'Richting', 'Geocode', 'KM', 'SpoorNummer', 'Trein_snelheid_in'],
        axis=1)

    # create an array that is true if Askwaliteit_asnummer is odd and false otherwise
    print('delete unnecessary rows')
    # mask = [row['Askwaliteit_asnummer'] % 2 == 1 for index, row in di_inputs_df.iterrows()]
    mask = di_inputs_df['Askwaliteit_asnummer'] % 2 == 1

    # delete all rows that have even Askwaliteit_asnummer
    di_inputs_df = di_inputs_df[mask]

    # convert speed from km/hr to m/s
    print('convert speed from km/hr to m/s')
    di_inputs_df['Trein_snelheid_uit'] = di_inputs_df['Trein_snelheid_uit'] * 0.2777

    # convert the mass from ton to kg, and multiply by 4 because we have 4 axles in each wagon
    print('calculate kg mass of each axes')
    di_inputs_df['Askwaliteit_aslast_kg'] = di_inputs_df['Askwaliteit_aslast_ton'] * 4000

    # conversions_dict helps in getting some aspects of the train based on its material properties
    conversions_dict = create_materials_conversion_dict(materials_properties_df)

    # get the available materials types within the material properties
    available_material_types = materials_properties_df["Voertuig_materieel_type"].tolist()

    # repalce all unavailable by the default values
    di_inputs_df['Voertuig_materieel_type'] = di_inputs_df['Voertuig_materieel_type'].apply(
        lambda x: x if x in available_material_types else 'Default'
    )

    # give the characteristics of each train (Wheel_type,Longitudinal_stiffness,Lateral_stiffness) 
    # based on the material properties by adding new columns to the dataframe
    # first map the material type to conversions_dict, then applies function extract_columns to the resulting dictionary
    print('Material type related calculations')
    di_inputs_df['Wheel_type'] = di_inputs_df['Voertuig_materieel_type'].map(conversions_dict).apply(
        lambda x: x['Wheel_type'])
    di_inputs_df['Longitudinal_stiffness'] = di_inputs_df['Voertuig_materieel_type'].map(conversions_dict).apply(
        lambda x: x['Longitudinal_stiffness'])
    di_inputs_df['Lateral_stiffness'] = di_inputs_df['Voertuig_materieel_type'].map(conversions_dict).apply(
        lambda x: x['Lateral_stiffness'])

    # Convert the dates to the number of days since the reference date
    di_inputs_df['days'] = (di_inputs_df['Datum'] - di_inputs_df.loc[0, 'Datum']).dt.days

    return di_inputs_df


def create_di_inputs(combined_raw_data_path, materials_properties_path, results_path):
    # read the pickled processed data
    with open(combined_raw_data_path, 'rb') as f:
        raw_df = dill.load(f)

    # create material properties dataframe
    materials_properties_df = pd.read_excel(materials_properties_path)

    # create damage index inputs data frame
    di_inputs_df = create_di_inputs_df(raw_df, materials_properties_df)

    with open(results_path, 'wb') as f:
        dill.dump(di_inputs_df, f)


def create_di_df(di_inputs_df, crack_init_model):
    # create damage index dataframe
    di_df = pd.DataFrame()

    # Use the processed data to calculate the RCF damage indices, and save it in the data frame
    di_df['damage_Index'] = crack_init_model.get_damage_indices(di_inputs_df)

    di_df['days'] = (di_inputs_df['Datum'] - di_inputs_df.loc[0, 'Datum']).dt.days

    # Calculate the damage index for periods of equal number of days
    di_df = create_var_per_day(di_df, variable_name='damage_Index')

    # #extrapolate data if there missing days in between
    di_df = extrapolate_df(di_df, x_name='days', y_name='damage_Index')

    di_df['accumulated_damage_Index'] = di_df['damage_Index'].cumsum()

    return di_df


def create_di_dfs(
        combined_raw_data_path, materials_properties_path, crack_init_coefficients_path, results_dir
):
    # read the pickled processed data
    with open(combined_raw_data_path, 'rb') as f:
        raw_df = dill.load(f)

    # create material properties dataframe
    materials_properties_df = pd.read_excel(materials_properties_path)

    # create damage index inputs data frame
    di_inputs_df = create_di_inputs_df(raw_df, materials_properties_df)

    for h in input_parameters["h"]:
        for radius in input_parameters["radius"]:
            for cant in input_parameters["cant"]:
                for coef in input_parameters["coeff"]:
                    # create crack initiation model
                    crack_init_model = CrackInitModel(
                        crack_init_coefficients_path,
                        h=h, radius=radius, cant=cant, cof=coef
                    )

                    # get results damage index data frame
                    di_df = create_di_df(di_inputs_df, crack_init_model)

                    filename = di_naming_pattern["write"].format(h, radius, cant, coef)
                    print(filename)

                    # save the results
                    with open(results_dir / filename, 'wb') as f:
                        dill.dump(di_df, f)





