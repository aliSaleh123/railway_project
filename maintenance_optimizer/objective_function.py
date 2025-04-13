from __future__ import annotations
from pathlib import Path
from logger import Logger
import copy
import numpy as np
from typing import List

from maintenance_model import EventsModel


def create_objective_function(
        events_model: EventsModel,
        num_iterations: int,
        print_interval=float("inf"),
        logger: Logger | None = None
):

    def objective_function(parameters: np.ndarray) -> List:
        if logger is not None:
            logger.sample_number += 1
            sample_number = logger.sample_number

        # Create local model
        events_model_local = copy.deepcopy(events_model)

        # Assign the parameters to the model
        events_model_local.rail.maintenance_policy = {
            "inspection_interval": 0.5,
            "grinding_interval": parameters[0],
            "milling_depth_threshold": parameters[1],
            "milling_portion_threshold": parameters[2],
            "replacement_depth_threshold": parameters[3]
        }

        # Initialize objectives
        hazard_durations = np.zeros(num_iterations)
        costs = {key: np.zeros(num_iterations) for key in events_model_local.rail.costs_calculator.value.keys()}

        # Solve

        for iter in range(num_iterations):

            if logger is not None:
                if iter % print_interval == 0:
                    logger.print(sample_number=sample_number, iteration_number=iter)

            # run the model
            events_model_local.run(end_time=30)

            hazard_durations[iter] = events_model_local.rail.hazard_duration

            for key in costs:
                costs[key][iter] = events_model_local.rail.costs_calculator.value[key]

            events_model_local.reset()

        # Structure the objectives
        # o1 = costs["inspection"].sum()
        o2 = costs["grinding"].sum() + costs["milling"].sum() + costs["replacement"].sum()
        o3 = hazard_durations.sum()

        return [o2, o3]

    return objective_function
