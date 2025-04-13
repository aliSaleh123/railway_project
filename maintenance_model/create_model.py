from .rail import Rail
from .events_model import EventsModel
from typing import Dict


def create_model(
        mgt_data_path, crack_init_times_path, transitions_path, affected_length_portion_path,
        maintenance_policy: Dict, save_results=False
):

    print("Creating the events model.")

    # Create the rail
    rail = Rail(
        mgt_data_path, crack_init_times_path, transitions_path, affected_length_portion_path,
        maintenance_policy, save_results
    )

    # create the events model
    model = EventsModel(rail)

    print("Events models created.")

    return model
