from maintenance_model import create_model
from config import PROCESSED_DATA_DIR, RESULTS_DIR
import matplotlib.pyplot as plt



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
        save_results=True
    )

    # run the model
    events_model.run(end_time=30)

    hazard_duration = events_model.rail.hazard_duration
    costs_details = events_model.rail.costs_calculator.value
    costs_total = events_model.rail.costs_calculator.total


    print(f"hazard duration: {hazard_duration}")
    for key in costs_details:
        print(f"Cost: {key}: {costs_details[key]}")
    print(f"Total Cost: {costs_total}")



    propagations = []
    for section in events_model.rail.sections:
        for part in section.parts:
            for crack in part.cracks:
                propagations.append(crack.get_depth_vs_time())

    for propagation in propagations:
        plt.plot(propagation[0], propagation[1])

    plt.xlabel("Time (Years)")
    plt.ylabel("Crack Depth (mm)")
    plt.title("The variation of crack depth versus time for all propagations.")
    plt.grid()
    plt.show()


if __name__ == "__main__":
    run()



