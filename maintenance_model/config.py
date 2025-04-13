# effect of maintenance actions in mm


epsilon = 1e-3  # for checking if cracks are less than 0

"""
The rail is composed of several sections and each section is divided into several parts.
"""

sections_config = [
    {"radius": 1500, "cant": 100, "length": 1000},
    {"radius": 1500, "cant": 100, "length": 1000},
    {"radius": 1500, "cant": 100, "length": 1000},
    ]

length_part = 10


grinding_config = {
    "price_shift": 20000,           # $/shift
    "time_shift": 8,                # hours
    "removal_rate": 0.035,          # mm/pass
    "removal_cost": 0.16,           # $/m when removing 0.035 mm
    "speed": 60000,                 # m/hr (between 40000 and 80000)
    "change_stones_time": 0.4,      # hr
    "change_stones_rate": 40000,    # m (every x meters the stones have to be changed)
    "num_cyclic_passes": 1          # number of grinding passes per grinding action
}

milling_config = {
    "price_shift": 18000,       # $/shift
    "time_shift": 10,           # hours
    "removal_rate": 1.8,        # mm/pass
    "speed": 750                # m/hr  (400 to 2000 m/hr)
}

inspection_config = {
    "speed": 80000,           # m/hour,
    "time_shift": 10,       # hours
    "shift_cost": 10000      # $
}

replacement_config = {
    "price_per_meter": 200,     # $/meter
    "price_shift": 20000,       # $/shift
    "time_shift": 10,           # hours
    "speed": 200,               # m/hr
}

maximum_removal_depth = 20

# Condition based on crack depth (mm)
conditions = {
    "light": 0,
    "medium": 10,
    "heavy": 20,
    "severe": 30
}

# The hazard factor corresponding to each condition
hazard_factor = {
    "light": 0,
    "medium": 0,
    "heavy": 1,
    "severe": 1
}




"""

use pareto optimization: https://www.youtube.com/watch?v=SL-u_7hIqjA
https://www.geeksforgeeks.org/non-dominated-sorting-genetic-algorithm-2-nsga-ii/



grinding is done to the whole rail
milling is done to the whole section
replacement is done per part

milling and grinding result in removal of surface even of good parts, so crack init time is restored for the whole section

if the removed depth from surface is greater than maximum_removal_depth, then replacement is necessary

"""