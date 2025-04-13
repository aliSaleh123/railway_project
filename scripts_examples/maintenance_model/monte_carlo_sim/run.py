from maintenance_model import create_model
from config import PROCESSED_DATA_DIR, RESULTS_DIR
import dill
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path

CURRENT_PATH = Path(__file__).resolve().parent


def run():

    # Creating the model
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

    hazard_durations = []
    costs = {key: [] for key in events_model.rail.costs_calculator.value.keys()}
    costs_total = []

    for iter in range(100):
        if iter % 10 == 0:
            print(iter)

        # run the model
        events_model.run(end_time=30)

        hazard_durations.append(events_model.rail.hazard_duration)

        for key in costs:
            costs[key].append(events_model.rail.costs_calculator.value[key])
        costs_total.append(events_model.rail.costs_calculator.total)

        events_model.reset()

    costs.update({"totals": costs_total})

    with open(CURRENT_PATH / "results" / "costs.pkl", 'wb') as f:
        dill.dump(costs, f)
    with open(CURRENT_PATH / "results" / "hazard_durations.pkl", 'wb') as f:
        dill.dump(hazard_durations, f)


def plot_distributions():

    with open(CURRENT_PATH / "results" / "costs.pkl", 'rb') as f:
        costs = dill.load(f)

    with open(CURRENT_PATH / "results" / "hazard_durations.pkl", 'rb') as f:
        hazard_durations = dill.load(f)

    sns.histplot(hazard_durations)
    plt.xlabel("Hazard factor")

    fig, axes = plt.subplots(5, 1, figsize=(10, 8))
    for i, (key, list_vals) in enumerate(costs.items()):
        sns.histplot(list_vals, ax=axes[i], kde=True, bins=200)
        # axes[i].set_title(key)
        axes[i].xaxis.set_major_formatter(mticker.ScalarFormatter(useMathText=False))  # Force plain numbers
        axes[i].ticklabel_format(style='plain', axis='x', useOffset=False)
        axes[i].set_ylabel(key)
    axes[-1].set_xlabel("Costs")
    plt.show()


if __name__ == "__main__":

    run()

    plot_distributions()




