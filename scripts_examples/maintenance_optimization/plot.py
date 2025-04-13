import dill
from pathlib import Path
# from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt


CURRENT_PATH = Path(__file__).resolve().parent


with open(CURRENT_PATH / "results" / "nsga_solver.pkl", "rb") as f:
    nsga_solver = dill.load(f)



all_populations = nsga_solver.all_populations
num_generations = len(nsga_solver.all_populations)





# plot the objectives (outputs)
plt.figure()
plt.title("Objectives")
plt.xlabel("Output 1")
plt.ylabel("Output 2")
for i in range(17, num_generations):
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
# #
# plt.show()









