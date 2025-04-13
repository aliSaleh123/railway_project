from scipy.stats import weibull_min
import dill
import matplotlib.pyplot as plt


def plot_transitions(transitions_path, figs_dir):
    with open(transitions_path, 'rb') as f:
        transitions = dill.load(f)

    for transition_type in transitions:
        for depth in transitions[transition_type]:
            data = transitions[transition_type][depth]
            if len(data) > 0:
                plt.hist(data, bins=10, density=True, alpha=0.5, color='gray', label='Data Histogram')
                plt.xlabel('Time to Failure')
                plt.ylabel('Probability Density')
                plt.legend()
                plt.grid()
                # plt.show()
                plt.savefig(figs_dir / f'{transition_type}_{depth}.png')
                plt.close()


def plot_test_simulations(results_dir):
    # Get all files that start with "results_123_" in the logs directory
    result_files = sorted(results_dir.glob("results_simtest_*.pkl"))

    ylim = 50

    for file in result_files:
        with open(file, 'rb') as f:
            data = dill.load(f)  # Assumes data is stored as [x_unified, y_mean, y_5th, y_95th]

        x_unified, y_mean, y_5th, y_95th = data

        name = file.name.split('results_simtest_')[1].split('.pkl')[0]

        # plt.fill_between(x_unified, y_5th, y_95th, alpha = 0.1)
        plt.plot(x_unified, y_mean, label=name)

        y_string = min(y_mean, key=lambda x: abs(x - ylim))
        x_string = x_unified[y_mean.tolist().index(y_string)]

        plt.text(x_string, y_string, f'{x_string:2.0f}, {y_string:2.0f}')

    plt.ylim(0, 50)
    plt.legend(title='backward limit')
    plt.show()
