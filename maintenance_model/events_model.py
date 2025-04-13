from abc import ABC, abstractmethod
from typing import List


class Event(ABC):

    @abstractmethod
    def update_time(self):
        pass

    @abstractmethod
    def fire(self):
        pass


class CrackInit(Event):
    def __init__(self, section, crack_init_dist):
        # the time of occurrence of the event
        self.time = float("inf")

        # reference to the events model
        self.events_model = None

        # the crack initiation should be related to a specific section
        self.section = section

        # the distribution of next crack initiation time
        self.dist = crack_init_dist

    def update_time(self):
        self.time = self.events_model.time + self.dist.sample()[0, 0]

    def fire(self):
        # initiate cracks in the section
        self.section.crack_initiation()

        # update the next crack init time
        self.update_time()


class Grinding(Event):
    def __init__(self, rail):
        # the time of occurrence of the event
        self.time = float("inf")

        # define the rail
        self.rail = rail

        # reference to the events model
        self.events_model = None

    def update_time(self):
        self.time = self.events_model.time + self.rail.maintenance_policy["grinding_interval"]

    def fire(self):
        # apply grinding for all sections of rail
        self.rail.apply_grinding()

        # update the next crack init time
        self.update_time()


class Inspection(Event):
    def __init__(self, rail):
        # the time of occurrence of the event
        self.time = float("inf")

        # define the rail
        self.rail = rail

        # reference to the events model
        self.events_model = None

    def update_time(self):
        self.time = self.events_model.time + self.rail.maintenance_policy["inspection_interval"]

    def fire(self):
        # apply grinding for all sections of rail
        self.rail.apply_inspection()

        # update the next crack init time
        self.update_time()


class EventsModel:
    events: List[Event]

    def __init__(self, rail: "Rail"):
        self.time = 0
        self.last_time = 0
        self.events = rail.events

        # dict for saving results
        self.results = {
            "states": [],
            "time": []
        }

        # link to the rail
        self.rail = rail
        self.rail.events_model = self

        # link the events to their model
        for event in self.events:
            event.events_model = self

        # initialize the first events times
        for event in self.events:
            event.update_time()

    def state_fn(self):
        if self.time > self.last_time:
            self.rail.update_condition()
        self.last_time = self.time

    def end_fn(self):
        self.state_fn()

        # # get the number of inspections per total time
        # number_inspections = np.floor(self.time / self.policy.inspection_interval)

    def run(self, end_time, save_results=False):

        while True:

            # loop on all events and get the minimum time
            self.time = min([event.time for event in self.events])

            if self.time > end_time:
                break

            # just before the change of state
            self.state_fn()

            for event in self.events:
                if event.time == self.time:
                    event.fire()

            if save_results:
                self.append_results()

        self.time = end_time
        self.end_fn()

    def reset(self):
        self.time = 0
        self.last_time = 0

        self.rail.reset()

        # initialize the first events times
        for event in self.events:
            event.update_time()

    def append_results(self):
        pass
