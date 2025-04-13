import pickle
import pandas as pd
import os
import dill


def combine_tonnage_data(tonnage_dir, combined_tonnage_path):
    files = [f for f in os.listdir(tonnage_dir) if os.path.isfile(os.path.join(tonnage_dir, f))]

    tonnage_dfs = []
    for file in files:
        # Read Excel sheet into DataFrame
        tonnage_dfs.append(
            pd.read_excel(tonnage_dir / file, usecols=['Spoortak_Identificatie', 'Dagtonnage_totaal', 'Periode_van'])
        )

    with open(combined_tonnage_path, 'wb') as f:
        dill.dump(tonnage_dfs, f)



