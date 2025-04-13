import dill

import numpy as np
from maintenance_model import create_model
from config import PROCESSED_DATA_DIR, RESULTS_DIR
from pathlib import Path
import matplotlib.pyplot as plt



def read_objectives(files_suffixes):
    objectives_list = []
    for i in files_suffixes:
        with open(CURRENT_PATH / "results" / "convergence_check_objectives_{}.pkl".format(i), "rb") as f:
            objectives_list.append(dill.load(f))

    objectives = {}
    for cost_part in objectives_list:
        for key in cost_part.keys():
            if key not in objectives:
                objectives[key] = []
            objectives[key].append(cost_part[key])

    objectives = {key: np.concatenate(objectives[key]) for key in objectives}
    return objectives



CURRENT_PATH = Path(__file__).resolve().parent

num_simulations = 300
print_interval = 10
results_suffix = 1

# Create the events model
events_model = create_model(
        mgt_data_path=RESULTS_DIR / "mgt_df.pkl",
        crack_init_times_path=RESULTS_DIR / "crack_init_times.pkl",
        transitions_path=RESULTS_DIR / "transitions" / "at_depth_limit_1.pkl",
        affected_length_portion_path=RESULTS_DIR / "affected_length_portion.pkl",
        maintenance_policy={
            "inspection_interval": 0.5,
            "grinding_interval": 0.2,
            "milling_depth_threshold": 3,
            "milling_portion_threshold": 1 / 100,
            "replacement_depth_threshold": 10
        },
        save_results=False
    )

# Initialize objectives
hazard_durations = np.zeros(num_simulations)
costs = {key: np.zeros(num_simulations) for key in events_model.rail.costs_calculator.value.keys()}


# Solve
for iter in range(num_simulations):

    if iter%print_interval == 0:
        print("iteration number {} within objective".format(iter))

    # run the model
    events_model.run(end_time=30)

    hazard_durations[iter] = events_model.rail.hazard_duration

    for key in costs:
        costs[key][iter] = events_model.rail.costs_calculator.value[key]

    events_model.reset()

objectives = costs
objectives.update({"hazard": hazard_durations})

# save the convergence check results
with open(CURRENT_PATH / "results" / "convergence_check_objectives_{}.pkl".format(results_suffix), "wb") as f:
    dill.dump(objectives, f)



# read costs from multiple results
objectives = read_objectives([0,1])

for key in objectives:
    plt.figure()

    average_vals = np.cumsum(objectives[key])/np.arange(1, len(objectives[key]) + 1)

    plt.title(key)
    plt.plot(average_vals)

plt.figure()
costs_total = np.stack([objectives["grinding"], objectives["milling"], objectives["replacement"]]).sum(axis=0)
average_vals = np.cumsum(costs_total)/np.arange(1, len(costs_total) + 1)
plt.title("total costs")
plt.plot(average_vals)

plt.show()