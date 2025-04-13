from nsga import NSGA
import numpy as np
import matplotlib.pyplot as plt


def objective_function(parameters):
    p1 = parameters[0]

    o1 = p1 ** 2
    o2 = (p1 - 2) ** 2

    return [o1, o2]


bounds = [(-10 ** 3, 10 ** 3)]


num_samples = 800
num_parents = 400
stds = [0.1]

nsga_solver = NSGA(
    num_samples=num_samples,
    num_parents=num_parents,
    bounds=bounds,
    stds=stds,
    objective_function=objective_function,
    save_results=True,
    run_parallel=False
)

nsga_solver.optimize(num_generations=30)

all_populations = nsga_solver.all_populations
num_generations = len(nsga_solver.all_populations)


# plot the objectives (outputs)
plt.figure()
plt.title("Objectives")
plt.xlabel("Output 1")
plt.ylabel("Output 2")
for i in range(num_generations):
    population = all_populations[i]
    if population.parents is not None:
        objectives = np.array(population.parents.objectives).T
        plt.scatter(objectives[0], objectives[1], label=i)
plt.legend(title="Generation")

# plot  the decision variables (inputs)
plt.figure()
plt.title("Decision variables")
plt.xlabel("Input 1")
plt.ylabel("Generation number")
for i in range(num_generations):
    population = all_populations[i]
    if population.parents is not None:
        samples = np.array(population.parents.samples).T
        ones = np.ones(samples.shape) * i
        plt.scatter(samples, ones, label=i)
plt.legend(title="Generation")

plt.show()