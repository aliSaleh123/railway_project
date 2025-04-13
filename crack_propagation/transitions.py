import dill
import numpy as np
import copy
from scipy.stats import weibull_min
import random
import matplotlib.pyplot as plt
from .config import CRACKS_DEPTHS_EC, CRACKS_DEPTHS_US, CRACKS_DEPTHS
last_index = len(CRACKS_DEPTHS) - 1


def calculate_average_and_percentiles(simulations, num_points=100):
    """
    Calculate the average, 5th, and 95th percentiles for multiple simulations.

    Parameters:
        simulations (list of lists): A list of simulations where each simulation
                                     contains two lists [x_values, y_values].
        num_points (int): Number of points for the unified x-axis.

    Returns:
        x_unified (numpy.ndarray): The unified x-axis.
        y_mean (numpy.ndarray): The average y values corresponding to the x_unified.
        y_5th (numpy.ndarray): The 5th percentile y values.
        y_95th (numpy.ndarray): The 95th percentile y values.
    """
    # Combine all x values to determine the min and max range
    all_x_values = np.concatenate([sim[0] for sim in simulations])
    x_min, x_max = np.min(all_x_values), np.max(all_x_values)

    # Create a unified x-axis
    x_unified = np.linspace(x_min, x_max, num_points)

    # Interpolate each simulation to the unified x-axis
    interpolated_y_values = []
    for sim in simulations:
        x, y = sim
        y_interp = np.interp(x_unified, x, y)
        interpolated_y_values.append(y_interp)

    # Convert to a numpy array for easier manipulation
    interpolated_y_values = np.array(interpolated_y_values)

    # Calculate statistics
    y_mean = np.mean(interpolated_y_values, axis=0)
    y_5th = np.percentile(interpolated_y_values, 5, axis=0)
    y_95th = np.percentile(interpolated_y_values, 95, axis=0)

    return x_unified, y_mean, y_5th, y_95th


def combine_dicts(dict_1, dict_2):
    dict_comb = {}
    for key in dict_1:
        dict_comb[key] = dict_1[key] + dict_2[key]
    return dict_comb


def create_transitions(
        processed_ec_sections_path, processed_us_sections_path, results_path,
        max_back_depth=1
):
    transitions_ec_data = sections_to_transitions(processed_ec_sections_path, max_back_depth)
    transitions_us_data = sections_to_transitions(processed_us_sections_path, max_back_depth)

    transitions_ec_processed = process_transitions(transitions_ec_data)
    transitions_us_processed = process_transitions(transitions_us_data)

    # combine transitions
    transitions = {
        "forward": combine_dicts(transitions_us_processed["forward"], transitions_ec_processed["forward"]),
        "inplace": combine_dicts(transitions_us_processed["inplace"], transitions_ec_processed["inplace"]),
        "backward": combine_dicts(transitions_us_processed["backward"], transitions_ec_processed["backward"]),
    }

    transitions = fill_missing_transitions(transitions)

    with open(results_path, 'wb') as f:
        dill.dump(transitions, f)


def sections_to_transitions(processed_sections_path, max_back_depth):
    with open(processed_sections_path, 'rb') as f:
        processed_sections = dill.load(f)

    transitions_data = {}
    for geo_code in processed_sections:
        for section in processed_sections[geo_code]:
            for crack in section.cracks:
                for key, value in crack.transitions_mgt.items():

                    if (key[0] == "unknown") | (key[1] == "unknown"):
                        continue

                    if key[0] > key[1]:
                        if key[0] - key[1] > max_back_depth:
                            continue

                    if not (key in transitions_data):
                        transitions_data[key] = []

                    transitions_data[key].append(value)

    return transitions_data


def process_transitions(transitions_data):
    def get_coefficients(depths):
        range_depths = depths[-1] - depths[0]
        coefficients = [(depths[i] - depths[i - 1]) / range_depths for i in range(1, len(depths))]
        return coefficients

    forward_transitions = {depth: [] for depth in CRACKS_DEPTHS}
    inplace_transitions = {depth: [] for depth in CRACKS_DEPTHS}
    backward_transitions = {depth: [] for depth in CRACKS_DEPTHS}
    for key in transitions_data:

        if (key[0] == "unknown") | (key[1] == "unknown"):
            continue

        if key[1] == key[0]:

            # if it is not the last depth of ec or the last depth of us then append in place transitions
            if (key[1] != CRACKS_DEPTHS_EC) & (key[1] != CRACKS_DEPTHS_US):
                # this will give a probability that the crack will not propagate
                inplace_transitions[key[1]] += transitions_data[key]

        else:
            # get depths values between these two measurements
            key_min = min(key)
            key_max = max(key)
            depths_steps = sorted(
                list(set([key_min] + [x for x in CRACKS_DEPTHS if key_min <= x <= key_max] + [key_max])))

            # get coefficients for dividing each mgt on multiple steps
            coefficients = get_coefficients(depths_steps)

            # discritize every mgt to multiple steps based on the coefficients
            dis_transitions = np.array([[trn * coeff for coeff in coefficients] for trn in transitions_data[key]])

            if key[1] > key[0]:
                # forward: fill the transitions of every depth, the last depth will not have a transition
                for i, depth in enumerate(depths_steps[0:-1]):
                    forward_transitions[depth] += dis_transitions[:, i].tolist()
            elif key[1] < key[0]:
                # backward: fill the transitions of every depth, the first depth will not have a transition
                for i, depth in enumerate(depths_steps[1:]):
                    backward_transitions[depth] += dis_transitions[:, i].tolist()

    transitions = {
        "forward": forward_transitions,
        "inplace": inplace_transitions,
        "backward": backward_transitions
    }

    return transitions


def fill_missing_transitions(transitions):
    # if any forward transitions list is empty for a specific depth, append all the transitions from previous depth
    for i in range(1, len(CRACKS_DEPTHS)):
        last_depth = CRACKS_DEPTHS[i - 1]
        current_depth = CRACKS_DEPTHS[i]

        # no forward transitions at a certain depth
        if len(transitions['forward'][current_depth]) == 0:

            # get correction coeff due to difference in depth jumps between transitions
            if current_depth == CRACKS_DEPTHS[-1]:
                coeff = 1
            else:
                next_depth = CRACKS_DEPTHS[i + 1]
                coeff = (next_depth - current_depth) / (current_depth - last_depth)

            if current_depth < CRACKS_DEPTHS[1]:
                coeff_back = 1
            else:
                last_last_depth = CRACKS_DEPTHS[i - 2]
                coeff_back = (current_depth - last_depth) / (last_depth - last_last_depth)

            # add the inplace and forward the transitions of the previous place
            transitions['forward'][current_depth] += [x * coeff for x in transitions['forward'][last_depth]]
            transitions['inplace'][current_depth] += transitions['inplace'][last_depth]
            transitions['backward'][current_depth] += [x * coeff_back for x in transitions['backward'][last_depth]]

    return transitions


def get_delta_depths():
    # get delta depth forward and backward at every crack depth
    delta_depths = {}

    # forward delta depth at each crack depth
    delta_depths['forward'] = {CRACKS_DEPTHS[i]: CRACKS_DEPTHS[i + 1] - CRACKS_DEPTHS[i] for i in
                               range(len(CRACKS_DEPTHS) - 1)}
    delta_depths['forward'][CRACKS_DEPTHS[-1]] = delta_depths['forward'][CRACKS_DEPTHS[-2]]

    # backward delta depth at each crack depth
    delta_depths['backward'] = {CRACKS_DEPTHS[i]: CRACKS_DEPTHS[i - 1] - CRACKS_DEPTHS[i] for i in
                                range(len(CRACKS_DEPTHS) - 1, 0, -1)}
    delta_depths['backward'][CRACKS_DEPTHS[0]] = None

    return delta_depths


def mgt_transitions_to_time_distributions(transitions, mgt_per_time):
    """

    Args:
        transitions (dict):
            transitions containing the mgt required to transition from crack depth to another
        mgt_per_time (float):
            for converting mgt to time

    Returns:
        distributions (dict):
            distributions to model the time required for transitioning from one crack depth to another
        transitions_out (dict):
            data used to fit the distributions (different from input if mgt_per_time!=1)
    """

    transitions_types = list(transitions.keys())
    distributions = {transition_type: {} for transition_type in transitions_types}
    transitions_out = {transition_type: {} for transition_type in transitions_types}
    for depth_key in CRACKS_DEPTHS:
        for transition_type in transitions_types:
            size_data = len(transitions[transition_type][depth_key])
            data = transitions[transition_type][depth_key].copy()

            # convert data from mgt to time
            data = [mgt/mgt_per_time for mgt in data]

            if size_data == 0:
                # return None
                def get_sample():
                    return None

            elif size_data == 1:
                # use the one sample
                value = data[0]

                def get_sample():
                    return value

            elif size_data == 2:
                # use a uniform distribution
                min_val, max_val = data[0], data[1]

                def get_sample():
                    return random.uniform(min_val, max_val)

            elif size_data <= 5:
                # use exponential distribution
                lambda_estimate = 1 / np.mean(data)

                def get_sample():
                    return np.random.exponential(scale=1 / lambda_estimate)

            else:
                # use weibull distribution
                shape, loc, scale = weibull_min.fit(data, floc=0)

                def get_sample():
                    return np.random.weibull(shape) * scale

            # save the distribution
            distributions[transition_type][depth_key] = get_sample
            transitions_out[transition_type][depth_key] = data

    return distributions, transitions_out


# class PropagationModel:
#     transitions_types = ['forward', 'inplace', 'backward']
#
#     def __init__(self, transitions, use_distributions=True):
#         self.transitions = transitions
#
#         # weights of transition types at each depth
#         self.transitions_types_weights = {
#             depth_key: [len(transitions[transition_type][depth_key]) for transition_type in self.transitions_types]
#             for depth_key in CRACKS_DEPTHS
#         }
#         # ensure that depth at 0 has no backward possibility
#         self.transitions_types_weights[0][2] = 0
#
#         # create a copy of transitions_types_weights in order to keep the original without change
#         self.transitions_types_weights_original = copy.deepcopy(self.transitions_types_weights)
#
#         # fit weibull distribution (MGT) for every depth at every of every transition type
#         self.distributions = mgt_transitions_to_time_distributions(transitions)
#
#         # create forward and backward delta depths at each crack depth
#         self.delta_depths = get_delta_depths()
#
#         # assign parameters
#         self.use_distributions = use_distributions
#
#     def transitions_control(self, inplace=True, backward=True, inplace_at_0=False):
#         # originally, all transitions are enabled
#         self.transitions_types_weights = copy.deepcopy(self.transitions_types_weights_original)
#
#         if inplace:
#             # only disable in place transitions at depth equals to 0
#             if not inplace_at_0:
#                 self.transitions_types_weights[0][1] = 0
#         else:
#             # disable all inplace transitions
#             for depth_key in CRACKS_DEPTHS:
#                 self.transitions_types_weights[depth_key][1] = 0
#
#         if not backward:
#             # disable all backward transitions
#             for depth_key in CRACKS_DEPTHS:
#                 self.transitions_types_weights[depth_key][2] = 0
#
#     def propagate(self, depth):
#
#         # get the depth key to retrieve dists from transitions (closest depth from the list)
#         depth_key = min(CRACKS_DEPTHS, key=lambda x: abs(x - depth))
#
#         # get the transition type
#         transition_type = random.choices(self.transitions_types, self.transitions_types_weights[depth_key])[0]
#
#         # get delta mgt to the next state
#         if self.use_distributions:
#             # using distributions
#             delta_mgt = self.distributions[transition_type][depth_key]()
#         else:
#             # using data
#             delta_mgt = random.choice(self.transitions[transition_type][depth_key])
#
#         # get delta depth
#         if transition_type == "inplace":
#             delta_depth = 0
#         else:
#             delta_depth = self.delta_depths[transition_type][depth_key]
#
#         return delta_mgt, delta_depth
class PropagationModel:
    transitions_types = ['forward', 'inplace', 'backward']

    def __init__(self, transitions, use_distributions=True, mgt_per_time=1):
        """

        Args:
            transitions:
            use_distributions:
            mgt_per_time (float):
                factor for converting distributions to time units
                the unit for this factor is (mgt/unit of time)
        """
        
        # weights of transition types at each depth
        self.transitions_types_weights = {
            depth_key: [len(transitions[transition_type][depth_key]) for transition_type in self.transitions_types]
            for depth_key in CRACKS_DEPTHS
        }
        # ensure that depth at 0 has no backward possibility
        self.transitions_types_weights[0][2] = 0

        # create a copy of transitions_types_weights in order to keep the original without change
        self.transitions_types_weights_original = copy.deepcopy(self.transitions_types_weights)

        # fit distributions (time units) for every depth at every of every transition type
        self.distributions, self.transitions = mgt_transitions_to_time_distributions(transitions, mgt_per_time)

        # create forward and backward delta depths at each crack depth
        self.delta_depths = get_delta_depths()

        # assign parameters
        self.use_distributions = use_distributions

    def transitions_control(self, inplace=True, backward=True, inplace_at_0=False):
        # originally, all transitions are enabled
        self.transitions_types_weights = copy.deepcopy(self.transitions_types_weights_original)

        if inplace:
            # only disable in place transitions at depth equals to 0
            if not inplace_at_0:
                self.transitions_types_weights[0][1] = 0
        else:
            # disable all inplace transitions
            for depth_key in CRACKS_DEPTHS:
                self.transitions_types_weights[depth_key][1] = 0

        if not backward:
            # disable all backward transitions
            for depth_key in CRACKS_DEPTHS:
                self.transitions_types_weights[depth_key][2] = 0

    def propagate(self, depth):

        # get the depth key to retrieve dists from transitions (closest depth from the list)
        depth_key = min(CRACKS_DEPTHS, key=lambda x: abs(x - depth))

        # get the transition type
        transition_type = random.choices(self.transitions_types, self.transitions_types_weights[depth_key])[0]

        # get delta time to the next state
        if self.use_distributions:
            # using distributions
            delta_time = self.distributions[transition_type][depth_key]()
        else:
            # using data
            delta_time = random.choice(self.transitions[transition_type][depth_key])

        # get delta depth
        if transition_type == "inplace":
            delta_depth = 0
        else:
            delta_depth = self.delta_depths[transition_type][depth_key]

        return delta_time, delta_depth

def one_simulation(propagation_model, mgt_end=50):
    depth_list = []
    mgt_list = []
    depth = 0
    mgt = 0
    while mgt < mgt_end:
        depth_list.append(depth)
        mgt_list.append(mgt)

        delta_mgt, delta_depth = propagation_model.propagate(depth)

        mgt += delta_mgt
        depth += delta_depth

    return mgt_list, depth_list


def test_propagation_model(
        transitions_path,
        results_dir,
        name='no_name',
        num_simulations=100,
        mgt_end=50,
):
    with open(transitions_path, 'rb') as f:
        transitions = dill.load(f)

    # create propagation model
    propagation_model = PropagationModel(transitions, use_distributions=False)

    # control the transitions to use
    propagation_model.transitions_control(inplace=True, backward=True, inplace_at_0=True)

    # gather mgt and depths of each simulations
    simulations = []
    for i in range(num_simulations):
        simulations.append(one_simulation(propagation_model, mgt_end))

    x_unified, y_mean, y_5th, y_95th = calculate_average_and_percentiles(simulations)

    with open(results_dir / f'results_simtest_{name}.pkl', 'wb') as f:
        dill.dump([x_unified, y_mean, y_5th, y_95th], f)


