import re
"""
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


# the crack initiation model is created for all combinations of the following values
input_parameters = {
    "h": [0.8, 1, 1.3, 1.5],
    "radius": [1000, 1500, 2000, 2500],
    "cant": [0, 50, 100, 150],
    "coeff": [0.01, 0.05, 0.15, 0.35]
}


num_format = r"(\d+(\.\d+)?)"

# pattern used for writing and reading damage index dataframes (di_df) results
di_naming_pattern = {
    "write": "di_h_{}_rad_{}_cant_{}_coef_{}.pkl",
    "read": re.compile("^di_h_{0}_rad_{0}_cant_{0}_coef_{0}.pkl$".format(num_format)),
    # "read": re.compile("^di_df_h_{0}_r_{0}.pkl$".format(num_format))
}

num_pattern = re.compile(num_format)


