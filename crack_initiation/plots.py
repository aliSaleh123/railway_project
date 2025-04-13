# -*- coding: utf-8 -*-
"""

@author: Ali Mohamad Saleh

"""
import re
import pickle
import dill
import numpy as np
import seaborn as sns
from scipy.stats import weibull_min
from sklearn.neighbors import KernelDensity
import matplotlib.pyplot as plt
from config import PROCESSED_DATA_DIR, LOGS_DIR, RESULTS_DIR
from .config import di_naming_pattern, num_pattern


def plot_DI_vs_days():
    file_path = PROCESSED_DATA_DIR / "usage" / "di_dfs.pkl"

    if file_path:

        with open(file_path, 'rb') as f:
            di_df = pickle.load(f)

        plt.figure(figsize=(8, 5))
        plt.scatter(di_df["days"], di_df["damage_Index"], s=2, color="blue", alpha=0.7)
        plt.title("Damage Index Over Time", fontsize=14)
        plt.xlabel("Days", fontsize=12)
        plt.ylabel("Damage Index", fontsize=12)
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.show()

    else:
        print(f"File: '{file_path}' does not exist.")


def plot_accumulated_DI_vs_days():
    di_dfs_dir = PROCESSED_DATA_DIR / "usage" / "di_dfs"

    result_files = [file.name for file in di_dfs_dir.iterdir() if di_naming_pattern["read"].match(file.name)]

    for filename in result_files:
        # get the variables values from the file name
        numbers = num_pattern.findall(filename)
        extracted_numbers = [float(num[0]) if '.' in num[0] else int(num[0]) for num in numbers]

        # h, radius = extracted_numbers
        h, radius, cant, coeff = extracted_numbers

        key = f"{radius}_{cant}"

        with open(di_dfs_dir / filename, 'rb') as f:
            di_df = pickle.load(f)

        plt.figure(num=key, figsize=(8, 5))
        plt.plot(di_df["days"], di_df["accumulated_damage_Index"], color="green", linewidth=1.5)
        plt.title("Accumulated Damage Index Over Time", fontsize=14)
        plt.xlabel("Days", fontsize=12)
        plt.ylabel("Accumulated Damage Index", fontsize=12)
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
    plt.show()


def plot_MGT_vs_days():
    file_path = RESULTS_DIR / "mgt_df.pkl"

    if file_path:

        with open(file_path, 'rb') as f:
            mgt_df = pickle.load(f)

        plt.figure(figsize=(8, 5))
        plt.scatter(mgt_df["days"], mgt_df["MGT"], s=2, color="orange", alpha=0.7)
        plt.title("MGT Over Time", fontsize=14)
        plt.xlabel("Days", fontsize=12)
        plt.ylabel("MGT", fontsize=12)
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.show()
    else:
        print(f"File: '{file_path}' does not exist.")


def plot_accumulated_MGT_vs_days():
    file_path = RESULTS_DIR / "mgt_df.pkl"

    if file_path:

        with open(file_path, 'rb') as f:
            mgt_df = pickle.load(f)

        x = mgt_df["days"]
        y = mgt_df["accumulated_MGT"]
        slope = np.sum(x * y) / np.sum(x ** 2)
        y2 = x*slope


        plt.figure(figsize=(8, 5))
        plt.plot(mgt_df["days"], mgt_df["accumulated_MGT"], color="red", linewidth=1.5, label = "original")
        plt.plot(mgt_df["days"], y2, color="blue", linewidth=1, label = "fitted")
        plt.title("Accumulated MGT Over Time", fontsize=14)
        plt.xlabel("Days", fontsize=12)
        plt.ylabel("Accumulated MGT", fontsize=12)
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.legend()
        plt.show()
    else:
        print(f"File: '{file_path}' does not exist.")


def plot_crack_init_dists(figures_path):
    crack_init_times_path = RESULTS_DIR / "crack_init_times.pkl"
    crack_init_dists_path = PROCESSED_DATA_DIR / "usage" / "crack_init_dists.pkl"

    with open(crack_init_times_path, 'rb') as f:
        crack_init_times = dill.load(f)

    with open(crack_init_dists_path, 'rb') as f:
        crack_init_dists = dill.load(f)

    for key in crack_init_times:
        sampled_data = crack_init_dists[key].sample(1000)
        data = np.array(crack_init_times[key])

        # get density of the distribution
        eval_points = np.linspace(sampled_data.min() - 10, sampled_data.max() + 10, 1000)[:, np.newaxis]
        kde_scores = crack_init_dists[key].score_samples(eval_points)  # Log density values
        density = np.exp(kde_scores)

        # Plot the results
        fig, (ax1, ax2) = plt.subplots(2,1, figsize=(16, 6), sharex=True)
        sns.histplot(data, bins=30, kde=False, stat='density', label='Data', color='skyblue', ax=ax1)
        ax1.plot(eval_points, density, label='KDE Fit', color='red')
        ax1.set_title('Data Distribution')
        ax1.set_xlabel('time to crack')
        ax1.set_ylabel('Density')
        ax1.legend()

        # Plot sampled_data on the second axis
        sns.histplot(sampled_data, bins=30, kde=False, stat='density', label='Sampled Data', color='lightgreen', ax=ax2)
        ax2.plot(eval_points, density, label='KDE Fit', color='red')
        ax2.set_title('Sampled Data Distribution')
        ax2.set_xlabel('time to crack')
        ax2.legend()

        # Add a title to the entire figure
        fig.suptitle(f'{key}', fontsize=16)

        # Save the figure
        plt.savefig(figures_path / f"radius_{key[0]}_cant_{key[1]}.png")



