# from __future__ import annotations

import dill
import numpy as np
from crack_propagation import PropagationModel
from crack_initiation import get_crack_init_dists
from typing import List
from .section import Section
from .events_model import Grinding, Inspection
from .costs import CostsCalculator
from .config import sections_config, length_part


class Rail:
    """
    Attributes:
        crack_init_dists (dict):
        propagation_model (PropagationModel):
        affected_length_portion (float):
        sections
        maintenance_policy
    """

    def __init__(
            self, mgt_data_path, crack_init_times_path, transitions_path, affected_length_portion_path,
            maintenance_policy, save_results=False
    ):
        """

        Args:
            mgt_data_path:
            crack_init_times_path:
            transitions_path:
            affected_length_portion_path:

        """

        # assigned once the events model is created
        self.events_model = None

        self.save_results = save_results

        # create the costs calculator
        self.costs_calculator = CostsCalculator(length_part=length_part)
        self.costs_calculator.rail = self

        # initialize the maintenance policy
        self.maintenance_policy = maintenance_policy

        # mgt versus time
        with open(mgt_data_path, 'rb') as f:
            mgt_df = dill.load(f)

        # get the slope of the MGT as a function of time (delta_mgt/delta_time)
        x = mgt_df["days"] / 365
        y = mgt_df["accumulated_MGT"]
        self.mgt_per_time = np.sum(x * y) / np.sum(x ** 2)  # (MGT/years)

        # crack initiation distributions for different radius and cant values
        with open(crack_init_times_path, 'rb') as f:
            crack_init_times = dill.load(f)

        # create the crack initiation distributions which allows sampling time (yrs.) to reach crack at different
        # conditions
        self.crack_init_dists = get_crack_init_dists(crack_init_times, data_factor=1 / 365)

        # ------------------------------------------------------------------------------------------
        # create propagation model based on the transitions
        with open(transitions_path, 'rb') as f:
            transitions = dill.load(f)

        # create propagation model
        self.propagation_model = PropagationModel(transitions, use_distributions=False, mgt_per_time=self.mgt_per_time)

        # control the transitions to use
        self.propagation_model.transitions_control(inplace=True, backward=True, inplace_at_0=True)

        # ------------------------------------------------------------------------------------------
        # get the affected length portion
        with (open(affected_length_portion_path, "rb")) as f:
            self.affected_length_portion = dill.load(f)

        # ------------------------------------------------------------------------------------------
        # Create sections
        self.sections = [
            Section(**section_config, length_part=length_part, rail=self) for section_config in sections_config
        ]

        # Calculate the length
        self.length = sum(section.length for section in self.sections)

        # ------------------------------------------------------------------------------------------
        # Create events
        self.grinding_event = Grinding(rail=self)
        self.inspection_event = Inspection(rail=self)
        self.events = \
            [section.crack_init_event for section in self.sections] + \
            [self.grinding_event, self.inspection_event]

    @property
    def hazard_duration(self):
        hazard_duration = 0
        for section in self.sections:
            for part in section.parts:
                hazard_duration += part.hazard_duration * part.length

        # normalize by the length of the rail
        hazard_duration /= self.length

        return hazard_duration

    def reset(self):
        self.costs_calculator.reset()
        for section in self.sections:
            section.reset()

    # def create_sections(self, sections: List["Section"]):
    #     self.sections += sections
    #     self.length = sum(section.length for section in sections)

    def get_crack_init_dist(self, radius, cant):
        keys = list(self.crack_init_dists.keys())

        radii = np.array([key[0] for key in keys])
        cants = np.array([key[1] for key in keys])

        # find the radius closest to the radii in keys
        radius_key = radii[np.argmin(np.abs(radius - radii))]
        cant_key = cants[np.argmin(np.abs(cant - cants))]

        return self.crack_init_dists[(radius_key, cant_key)]

    def apply_grinding(self):
        for section in self.sections:
            section.grinding()

        # assign grinding costs
        self.costs_calculator.pay_grinding()

    def apply_inspection(self):
        # after inspection, the condition of cracks become known, so it is possible to take corrective decision
        for section in self.sections:
            # check if any corrective decisions will be taken for the section
            section.corrective_decision()

        # assign inspection costs
        self.costs_calculator.pay_inspection()

    def update_condition(self):
        # todo the main function to be called before every change of state

        # update the conditions of all initiated cracks within the parts
        for section in self.sections:
            for part in section.parts:
                if part.condition != "good":
                    part.propagate_crack()

