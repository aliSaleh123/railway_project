# -*- coding: utf-8 -*-
"""

@author: Ali Mohamad Saleh

"""
from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from typing import List, Callable, Union, Any, Tuple
from logger import Logger
import copy
import dill
from pathlib import Path


def linear_fn_factory(x1, y1, x2, y2):
    if y1 == y2:
        return lambda x: y1
    else:
        a = (y2 - y1) / (x2 - x1)
        b = y1 - a * x1

        return lambda x: a * x + b


def get_indices_within_bounds(samples: np.ndarray, bounds: List):
    check_bound = [1 if bound[0] != bound[1] else 0 for bound in bounds]

    condition = np.ones(samples.shape[0], dtype=bool)
    for i in range(len(bounds)):
        if check_bound[i]:
            condition &= (samples[:, i] >= bounds[i][0]) & (samples[:, i] <= bounds[i][1])

    return np.where(condition)[0]


def ensure_bounds(
        samples: np.ndarray,
        bounds: List,
        method: str = 'boundary'
):
    """
    if there are samples outside the bounds, remove them and locate them inside the bounds

    method:
        random: place samples randomly within bounds
        boundary: place samples on the closest boundary

    """

    accepted_indices = get_indices_within_bounds(samples, bounds)
    num_rejected = samples.shape[0] - len(accepted_indices)

    # filter samples that are outside the bounds
    if num_rejected > 0:

        if method == 'random':

            # filter new samples
            samples = samples[accepted_indices]

            # propose samples instead of the rejected ones
            additional_samples = random_samples(num_rejected, bounds)

            # combine
            samples = np.vstack((samples, additional_samples))

        elif method == 'boundary':
            for i, (lower, upper) in enumerate(bounds):
                samples[:, i] = np.clip(samples[:, i], lower, upper)

    return samples


def random_samples(num_samples, bounds):
    """Generate Samples Based on Uniform Distribution"""
    num_dimensions = len(bounds)
    samples = np.zeros((num_samples, num_dimensions))

    for i, (lower, upper) in enumerate(bounds):
        samples[:, i] = np.random.uniform(lower, upper, num_samples)

    return samples


class Parents:
    def __init__(self):
        self.samples = []
        self.objectives = []
        self.ranks = []  # based on front number
        self.crowding_distances = []


class Children:
    def __init__(self):
        self.samples = []
        self.objectives = []


class Population:
    def __init__(
            self,
            samples: list,
            objectives: list,
            parents: Parents | None = None,
            children: Children | None = None
    ):
        self.samples = samples
        self.objectives = objectives

        self.parents = parents
        self.children = children

    @classmethod
    def from_parents_and_children(cls, parents: Parents, children: Children):
        samples = parents.samples + children.samples
        objectives = parents.objectives + children.objectives
        return cls(samples, objectives, parents, children)



class NSGA():
    def __init__(
            self,
            num_samples: int,
            num_parents: int,
            bounds: List[Tuple[float, float]],
            stds: List[Union[float, Callable]],
            objective_function: Callable,
            save_results: bool = False,
            run_parallel: bool = False,
            logger: Logger | None = None
    ):
        self.bounds = bounds
        self._stds = stds
        self.objective_function = objective_function

        self.num_parameters = len(bounds)
        self.num_samples = num_samples
        self.num_parents = num_parents

        self.population: Population = None



        self.save_results = save_results

        self.all_populations: List[Population] = []

        self.run_parallel = run_parallel

        self.logger = logger

        print("NSGA solver created")

    def get_stds(self, gen_idx):
        return [std(gen_idx) if callable(std) else std for std in self._stds]

    @staticmethod
    def fast_non_dominated_sort(objectives: List[List[float]]) -> list:
        """
        Perform fast non-dominated sorting on an array of objectives.

        Args:
            - objectives:
                A 2D NumPy array where each row represents a solution, and each column
                represents an objective function value.

        Returns:
            list of list: A list of fronts, where each front is a list of indices corresponding
            to the solutions in that front.
        """
        population_size = len(objectives)
        S = [[] for _ in range(population_size)]  # Dominance sets
        n = [0] * population_size  # Dominated counts
        fronts = [[]]  # List of Pareto fronts

        # Step 1: Calculate dominance relationships
        for p in range(population_size):
            for q in range(population_size):
                if all(obj_p <= obj_q for obj_p, obj_q in zip(objectives[p], objectives[q])) and any(
                        obj_p < obj_q for obj_p, obj_q in zip(objectives[p], objectives[q])):
                    S[p].append(q)
                elif all(obj_q <= obj_p for obj_p, obj_q in zip(objectives[p], objectives[q])) and any(
                        obj_q < obj_p for obj_p, obj_q in zip(objectives[p], objectives[q])):
                    n[p] += 1

            # If no solutions dominate p, it belongs to the first front
            if n[p] == 0:
                fronts[0].append(p)

        # Step 2: Identify subsequent fronts
        i = 0
        while fronts[i]:
            next_front = []
            for p in fronts[i]:
                for q in S[p]:
                    n[q] -= 1
                    if n[q] == 0:
                        next_front.append(q)
            i += 1
            fronts.append(next_front)

        # Remove the last empty front
        fronts.pop()

        return fronts

    @staticmethod
    def calculate_crowding_distance(objectives):
        """
        Calculate the crowding distance for a set of solutions based on their objective values.

        Parameters:
        - objectives: A 2D NumPy array where each row represents a solution, and each column
                      represents an objective function value.

        Returns:
        - distances: A 1D NumPy array containing the crowding distance for each solution.
        """

        objectives = np.array(objectives)

        num_samples, num_objectives = objectives.shape

        # Initialize distances to zero
        distances = np.zeros((num_objectives, len(objectives)))

        # For each objective, calculate crowding distances
        for i in range(num_objectives):
            # Sort solutions based on the current objective
            sorted_indices = np.argsort(objectives[:, i])
            sorted_objectives = objectives[sorted_indices, i]

            # Set the boundary solutions to have an infinite distance
            distances[i, sorted_indices[0]] = np.inf
            distances[i, sorted_indices[-1]] = np.inf

            # Calculate normalized objective range
            objective_range = sorted_objectives[-1] - sorted_objectives[0]
            if objective_range == 0:
                continue

            # Compute distances for interior points
            for j in range(1, num_samples - 1):
                prev_obj = sorted_objectives[j - 1]
                next_obj = sorted_objectives[j + 1]
                distances[i, sorted_indices[j]] += (next_obj - prev_obj) / objective_range

        distances = distances.sum(axis=0)

        return distances

    def get_children_indexes(self, parents):

        # Selection of parents for mutation
        # Select randomly two indices of the parents, do this iteratively to form num_parents
        # Then perform binary tournament based on ranks and crowding_distances to choose
        # one of the chosen samples
        indexes = np.random.randint(0, len(parents.ranks), size=(self.num_parents, 2))
        children_indexes = []

        for indx1, indx2 in indexes:
            # compare two samples
            if parents.ranks[indx1] < parents.ranks[indx2]:
                selected_index = indx1
            elif parents.ranks[indx1] > parents.ranks[indx2]:
                selected_index = indx2
            else:
                # based on crowding distance
                if parents.crowding_distances[indx1] > parents.crowding_distances[indx2]:
                    selected_index = indx1
                else:
                    selected_index = indx2

            children_indexes.append(selected_index)

        return children_indexes

    def select_parents(self, fronts):
        # Create the parents set
        len_parents_set = 0
        parents = Parents()

        for front_id, samples_indexes in enumerate(fronts):
            samples_front = [self.population.samples[index] for index in samples_indexes]
            objectives_front = [self.population.objectives[index] for index in samples_indexes]
            crowding_distances_front = list(self.calculate_crowding_distance(objectives_front))

            len_front = len(samples_indexes)

            if len_parents_set + len_front <= self.num_parents:

                # add samples to parents set
                parents.samples += samples_front

                # add objectives
                parents.objectives += objectives_front

                # calculate the crowding distance of this set
                parents.crowding_distances += crowding_distances_front

                # assign front number for every parent
                parents.ranks += [front_id for _ in range(len_front)]

                len_parents_set += len_front

            else:
                # get the number of vacant parents
                num_vacant_parents = self.num_parents - len_parents_set

                # Combine the indexes and distances into tuples
                combined = list(zip(samples_indexes, crowding_distances_front))

                # Sort by crowding distances in descending order
                combined_sorted = sorted(combined, key=lambda x: x[1], reverse=True)

                # Extract the top m indexes
                selected_indexes = [x[0] for x in combined_sorted[:num_vacant_parents]]

                # add samples to parents set
                parents.samples += [self.population.samples[index] for index in selected_indexes]

                # add objectives
                parents.objectives += [self.population.objectives[index] for index in selected_indexes]

                # calculate the crowding distance of this set
                parents.crowding_distances += [x[1] for x in combined_sorted[:num_vacant_parents]]

                # assign front number for every parent
                parents.ranks += [front_id for _ in range(num_vacant_parents)]

                break

        return parents

    def evaluate_samples(self, samples):
        if self.logger is not None:
            self.logger.sample_number = 0
        if self.run_parallel:
            with ThreadPoolExecutor() as executor:
                objectives = list(executor.map(self.objective_function, samples))
        else:
            objectives = [self.objective_function(parameters) for parameters in samples]

        return objectives

    def optimize(
            self,
            num_generations: int,
            write_results_interval=float("inf"),
            results_folder: Path | None = None
    ):
        if results_folder:
            if not results_folder.exists():
                results_folder.mkdir(parents=True)


        if self.population is None:
            # initialize the population
            samples = list(random_samples(num_samples=self.num_samples, bounds=self.bounds))

            # evaluate the objective function
            objectives = self.evaluate_samples(samples)
            self.population = Population(samples, objectives)

            if self.save_results:
                self.all_populations.append(self.population)

        for gen_idx in range(num_generations):

            if self.logger is not None:
                self.logger.generation_number = gen_idx + 1
                self.logger.sample_number = 0
                self.logger.print()

            fronts = self.fast_non_dominated_sort(self.population.objectives)

            # select parents from the current population
            parents = self.select_parents(fronts)

            # Create Children
            children_indexes = self.get_children_indexes(parents)

            children = Children()
            children.samples = np.array([parents.samples[index] for index in children_indexes])
            children.samples = \
                children.samples + \
                np.random.normal(loc=0, scale=self.get_stds(gen_idx), size=children.samples.shape)

            # todo check if this is working properly
            children.samples = list(ensure_bounds(children.samples, self.bounds))

            # Evaluate new samples
            children.objectives = self.evaluate_samples(children.samples)

            # create a new population by combining children and parents
            self.population = Population.from_parents_and_children(
                copy.deepcopy(parents),
                copy.deepcopy(children),
            )

            if self.save_results:
                self.all_populations.append(self.population)

            if gen_idx % write_results_interval == 0:
                if results_folder:
                    with open(results_folder / f"nsga_solver_{gen_idx}.pkl", "wb") as f:
                        dill.dump(self, f)

        if self.logger is not None:
            self.logger.generation_number = 0

    def reset(self):
        self.population = None
        self.all_populations = None
