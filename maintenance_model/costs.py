from .config import milling_config, grinding_config, inspection_config, replacement_config


class MillingCost:
    def __init__(self, price_shift, time_shift, removal_rate, speed):
        """

        Args:
            price_shift:
            time_shift:
            removal_rate:
            speed:
        """
        # constant related to the operation
        self.price_shift = price_shift  # $
        self.time_shift = time_shift  # hours
        self.removal_rate = removal_rate  # mm/pass
        self.speed = speed  # m/hr  (400 to 2000 m/hr)

        self.cost_per_meter = self._get_costs_detailed(length=1)

    def _get_costs_detailed(self, length):
        # the time required to finish the milling
        required_time = length / self.speed

        # the number of shifts required to finish milling
        required_shifts = required_time / self.time_shift

        # total maintenance costs
        cost = required_shifts * self.price_shift

        return cost


class GridingCost:

    def __init__(
            self,
            price_shift, time_shift, removal_rate, removal_cost, speed, change_stones_time, change_stones_rate,
            num_cyclic_passes
    ):
        """

        Args:
            price_shift:
            time_shift:
            removal_rate:
            removal_cost:
            speed:
            change_stones_time:
            change_stones_rate:
            num_cyclic_passes:
        """
        # constant related to the operation
        self.price_shift = price_shift  # $/shift
        self.time_shift = time_shift  # hours
        self.removal_rate = removal_rate  # mm/pass
        self.removal_cost = removal_cost  # $/m when removing 0.035 mm

        self.speed = speed  # m/hr (between 40000 and 80000)
        self.change_stones_time = change_stones_time  # hr
        self.change_stones_rate = change_stones_rate  # m

        self.num_cyclic_passes = num_cyclic_passes

        # time for finishing one grinding cycle
        operation_time = change_stones_rate / speed

        # time for finishing one grinding cycle including the time to change the grinding stones
        time_per_grinding_cycle = operation_time + change_stones_time  # time per grinding cycle in hours

        # grinding speed is calculated as the time to finish a grinding cycle divided by the length finished in a
        # grinding cycle
        self.grinding_speed = change_stones_rate / time_per_grinding_cycle  # m/hr

        # get costs per meter of grinding
        self.cost_per_meter = self._get_costs_detailed(length=1)

    def _get_costs_detailed(self, length):
        # The total time required to finish the whole branch
        total_time = length * self.num_cyclic_passes / self.grinding_speed

        # The number of required shifts for the rail branch
        num_required_shifts = total_time / self.time_shift

        # costs of the shifts
        shifts_costs = num_required_shifts * self.price_shift

        # costs of the grinding actions
        grinding_costs = length * self.num_cyclic_passes * self.removal_cost

        total_cost = shifts_costs + grinding_costs

        return total_cost


class InspectionCosts:

    def __init__(self, speed, time_shift, shift_cost):
        self.speed = speed
        self.time_shift = time_shift
        self.shift_cost = shift_cost

        self.cost_per_meter = self._get_cost(length=1)

    def _get_cost(self, length):

        required_time = length / self.speed

        required_shifts = required_time / self.time_shift

        costs = required_shifts * self.shift_cost

        return costs


class ReplacementCosts:

    def __init__(self, price_per_meter, price_shift, time_shift, speed):
        self.price_per_meter = price_per_meter
        self.price_shift = price_shift
        self.time_shift = time_shift
        self.speed = speed

        self.cost_per_meter = self._get_cost(length=1)

    def _get_cost(self, length):
        required_time = length / self.speed

        required_shifts = required_time / self.time_shift

        shifts_costs = required_shifts * self.price_shift

        replacement_costs = length * self.price_per_meter

        return shifts_costs + replacement_costs


class CostsCalculator:
    def __init__(self, length_part):
        self.grinding_costs = GridingCost(**grinding_config)
        self.milling_costs = MillingCost(**milling_config)
        self.replacement_costs = ReplacementCosts(**replacement_config)  # per length replaced
        self.inspection_costs = InspectionCosts(**inspection_config)  # per length inspected

        self.replacement_cost_per_part = self.replacement_costs.cost_per_meter * length_part

        # assigned once passed to rail
        self.rail = None

        self.value = {
            "grinding": 0,
            "milling": 0,
            "replacement": 0,
            "inspection": 0
        }

    @property
    def total(self):
        return sum(self.value.values())

    def print_costs_per_meter(self):

        print(f"Inspection Costs Per Meter: {self.inspection_costs.cost_per_meter:2.2e}")
        print(f"Grinding Costs Per Meter: {self.grinding_costs.cost_per_meter:2.2e}")
        print(f"Milling Costs Per Meter: {self.milling_costs.cost_per_meter:2.2e}")
        print(f"Replacement Costs Per Meter: {self.replacement_costs.cost_per_meter:2.2e}")

    def reset(self):
        for key in self.value:
            self.value[key] = 0

    def pay_part_replacement(self):
        # done per part
        self.value["replacement"] += self.replacement_cost_per_part

    def pay_grinding(self):
        # done the whole rail
        self.value["grinding"] += self.grinding_costs.cost_per_meter * self.rail.length

    def pay_milling(self, number_passes, length):
        # done per section
        self.value["milling"] += self.milling_costs.cost_per_meter * length * number_passes

    def pay_inspection(self):
        # done per whole rail
        self.value["inspection"] += self.rail.length * self.inspection_costs.cost_per_meter

# if __name__ == "__main__":
#     import numpy as np
#     import matplotlib.pyplot as plt
#
#     grinding_cost = GridingCost(**grinding_config)
#     lengths = np.linspace(0, 1000000, 100)
#     costs = [grinding_cost.get_cost(l) for l in lengths]
#     costs2 = [grinding_cost._get_costs_detailed(l) for l in lengths]
#     plt.plot(lengths, costs)
#     plt.plot(lengths, costs2, 'x')
#
#     plt.show()
