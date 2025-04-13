import dill

from maintenance_optimizer import NSGA, create_objective_function
from maintenance_optimizer import linear_fn_factory as lff
from maintenance_model import create_model
from config import PROCESSED_DATA_DIR, RESULTS_DIR
from pathlib import Path
from logger import Logger

CURRENT_PATH = Path(__file__).resolve().parent

# todo run and check the results


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
            "milling_portion_threshold": 2 / 100,
            "replacement_depth_threshold": 10
        },
        save_results=False
    )

logger = Logger()



# bounds: inspection_interval, grinding_interval, milling_depth_threshold, milling_portion_threshold,
# milling_portion_threshold, replacement_depth_threshold
bounds = [
    # (0.1, 1),                 # inspection_interval (yrs.)
    (0.1, 1),                   # grinding_interval (yrs.)
    (0.1, 35),                  # milling_depth_threshold (mm)
    (0.05/100, 40/100),         # milling_portion_threshold
    (0.1, 35),                  # replacement_depth_threshold (mm)
]


num_generations = 40
num_samples = 400
num_parents = 200
num_iterations = 500

stds = [
    # 0.1,    # inspection_interval (yrs.)
    lff(0, 0.2, num_generations-1, 0.01),           # grinding_interval (yrs.)
    lff(0, 2, num_generations-1, 0.1),             # milling_depth_threshold (mm)
    lff(0, 2/100, num_generations-1, 0.5/100),     # milling_portion_threshold
    lff(0, 2, num_generations-1, 0.1)            # replacement_depth_threshold (mm)
]


# Create the objective function
objective_function = create_objective_function(events_model, num_iterations=num_iterations, print_interval=5, logger=logger)

# Define the optimizer
nsga_solver = NSGA(
    num_samples=num_samples,
    num_parents=num_parents,
    bounds=bounds,
    stds=stds,
    objective_function=objective_function,
    save_results=True,
    run_parallel=True,
    logger=logger
)

# Optimize
nsga_solver.optimize(
    num_generations=num_generations,
    write_results_interval=1,
    results_folder=CURRENT_PATH / "results" / 'optimization'
)


with open(CURRENT_PATH / "results" / "nsga_solver.pkl", "wb") as f:
    dill.dump(nsga_solver, f)


