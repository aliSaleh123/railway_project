import random
import numpy as np

from .config import grinding_config, milling_config, maximum_removal_depth, epsilon, conditions, hazard_factor
from .events_model import CrackInit

grinding_depth = grinding_config["removal_rate"] * grinding_config["num_cyclic_passes"]


class Section:
    def __init__(self, radius, cant, length, length_part, rail: "Rail"):
        self.rail = rail
        self.radius = radius
        self.cant = cant
        self.length = length

        # get the crack initiation distribution for the section
        crack_init_dist = rail.get_crack_init_dist(radius, cant)

        # create the crack initiation event of the section
        self.crack_init_event = CrackInit(self, crack_init_dist)

        # get the number of parts in the section
        self.num_parts = int(self.length / length_part)

        # get the actual length of each part
        self.length_part = self.length / self.num_parts

        # initialize the parts list
        self.parts = [Part(self, rail, self.length_part) for _ in range(self.num_parts)]

        # the number of parts that are affected each time cracks initiate in the section
        self.num_newly_affected_parts = int(self.rail.affected_length_portion * self.num_parts)

    def reset(self):
        for part in self.parts:
            part.reset()

    def crack_initiation(self):
        # get a list of parts without cracks
        parts_without_cracks = [part for part in self.parts if part.condition == "good"]

        # select number of parts randomly to initiate cracks
        parts_to_crack = random.sample(
            parts_without_cracks,
            min(self.num_newly_affected_parts, len(parts_without_cracks))
        )

        # initiate cracks within the parts
        for part in parts_to_crack:
            part.initiate_crack()

    def corrective_decision(self):

        # get the parts to be milled. These are the ones with crack that is higher than
        # milling depth threshold and lower than replacement depth threshold
        parts_to_mill = [
            part for part in self.parts
            if self.rail.maintenance_policy["milling_depth_threshold"] <= part.crack_depth <=
               self.rail.maintenance_policy["replacement_depth_threshold"]
        ]

        # get the parts to be replaced, which are the ones with cracks greater than replacement_depth_threshold
        parts_to_replace = [
            part for part in self.parts if
            part.crack_depth > self.rail.maintenance_policy["replacement_depth_threshold"]
        ]

        # check if enough parts require milling (milling portion threshold is met)
        # get the length of parts that require milling over the length of the whole section
        if len(parts_to_mill)*self.length_part/self.length > self.rail.maintenance_policy["milling_portion_threshold"]:

            # all the crack initiation times are reset after any removal of the surface of the rail
            self.crack_init_event.update_time()

            # get the required milling depth based on the maximum crack depth of the part that required milling
            required_milling_depth = max([part.crack_depth for part in parts_to_mill])

            # the number of passes required
            num_milling_passes = np.ceil(required_milling_depth / milling_config["removal_rate"])

            # pay the costs of milling the section
            self.rail.costs_calculator.pay_milling(num_milling_passes, self.length)

            # apply milling actions to the required parts
            for part in self.parts:
                part.milling(num_milling_passes)

        # check if any part require milling
        if len(parts_to_replace) > 0:

            # apply replacement actions to the parts
            for part in parts_to_replace:
                part.replacement()

    def grinding(self):
        # all the crack initiation times are reset after any removal of the surface of the rail
        self.crack_init_event.update_time()

        # repair parts based on the repair method
        for part in self.parts:
            part.grinding()


class Part:
    def __init__(self, section: Section, rail: "Rail", length: float):
        self.rail = rail
        self.section = section
        self.length = length
        self.crack_depth = 0
        self.removed_depth = 0
        self.condition = "good"

        # the duration at which the part was in severe condition
        self.hazard_duration = 0

        # the costs encountered
        self.costs = 0

        # in case save results
        # for saving the all the cracks propagations this section has encountered
        self.cracks = []

    def reset(self):
        self.crack_depth = 0
        self.removed_depth = 0
        self.condition = "good"
        self.hazard_duration = 0
        self.costs = 0
        self.cracks = []

    def initiate_crack(self):
        self.condition = "light"

        if self.rail.save_results:
            self.cracks.append(Crack(self))

    def propagate_crack(self):

        time_reached = self.rail.events_model.last_time
        time_final = self.rail.events_model.time

        depth_reached = self.crack_depth

        while True:
            d_time, d_depth = self.rail.propagation_model.propagate(depth_reached)

            if time_reached + d_time > time_final:
                d_time_actual = time_final - time_reached

                # change d_depth and d_time based on the actual time passed
                d_depth = d_depth * d_time_actual / d_time
                d_time = d_time_actual

            # update time and depth reached
            time_reached += d_time
            depth_reached = max(depth_reached + d_depth, 0)
            # if d_depth = -0.3 and depth_reached was greater than 0 but less than 0.3:
            #    depth_reached can be less than 0, for this it is bounded with the 0

            # update the reached condition
            if depth_reached >= conditions["severe"]:
                condition_reached = "severe"
            elif depth_reached >= conditions["heavy"]:
                condition_reached = "heavy"
            elif depth_reached >= conditions["medium"]:
                condition_reached = "medium"
            elif depth_reached >= conditions["light"]:
                condition_reached = "light"

            # update the hazard duration of the part
            self.hazard_duration += d_time * hazard_factor[condition_reached] * depth_reached

            # save the reached conditions in case conditions are being saved
            if self.rail.save_results:
                self.cracks[-1].new_propagation(time_reached, depth_reached, condition_reached)

            if time_reached >= time_final:
                # update the depth of the crack
                self.crack_depth = depth_reached
                self.condition = condition_reached
                break

    def grinding(self):

        self.removed_depth += grinding_depth

        if self.removed_depth > maximum_removal_depth:
            self.replacement()

        if self.condition != "good":
            self.crack_depth -= grinding_depth

            if self.crack_depth - epsilon < 0:
                self.condition = "good"
                self.crack_depth = 0

    def milling(self, num_milling_passes):
        # after corrective action, always all parts will be cracks free
        self.condition = "good"
        self.crack_depth = 0
        self.removed_depth += milling_config["removal_rate"] * num_milling_passes

        if self.removed_depth > maximum_removal_depth:
            self.replacement()

    def replacement(self):
        # after corrective action, always all parts will be cracks free
        self.condition = "good"
        self.crack_depth = 0
        self.removed_depth = 0

        # pay the costs of replacing the part
        self.rail.costs_calculator.pay_part_replacement()


class Crack:
    """
    This class is only used to save the results of the cracks propagations within the parts
    """

    def __init__(self, part):
        self.part = part
        self.section = part.section
        self.rail = part.section.rail

        # the current time of the events model
        self.initiation_time = self.rail.events_model.time

        self.time_vals = [self.initiation_time]
        self.depth_vals = [0]
        self.conditions = ["good"]

    def new_propagation(self, time, depth, condition):
        self.time_vals.append(time)
        self.depth_vals.append(depth)
        self.conditions.append(condition)

    def get_depth_vs_time(self):
        return np.array([self.time_vals, self.depth_vals])

    def get_depth_vs_abs_time(self):
        d_vs_t = np.array([self.time_vals, self.depth_vals])
        d_vs_t[0] = d_vs_t[0] - self.initiation_time
        return d_vs_t

    def get_depth_vs_abs_mgt(self):
        d_vs_mgt = np.array([self.time_vals, self.depth_vals])
        d_vs_mgt[0] = (d_vs_mgt[0] - self.initiation_time) * self.rail.mgt_per_time
        return d_vs_mgt
