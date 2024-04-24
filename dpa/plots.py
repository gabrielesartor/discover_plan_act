import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    ab_results = np.load("data/02_two_floors_bolt/ACTION_BABBLING/results_ACTION_BABBLING.npy")
    gb_results = np.load("data/02_two_floors_bolt/GOAL_BABBLING/results_GOAL_BABBLING.npy")
    db_results = np.load("data/02_two_floors_bolt/DISTANCE_BABBLING/results_DISTANCE_BABBLING.npy")

    ab_mean = np.mean(ab_results, axis = 0)
    gb_mean = np.mean(gb_results, axis = 0)
    db_mean = np.mean(db_results, axis = 0)

    fig = plt.figure()
    x = np.arange(len(ab_mean))
    yerr = np.linspace(0.01, 0.02, 15)

    plt.errorbar(x, ab_mean, yerr=yerr, label='AB')

    plt.errorbar(x, gb_mean, yerr=yerr, label='GB')

    plt.errorbar(x, db_mean, yerr=yerr, label='DB')

    # upperlimits = [True, False] * 5
    # lowerlimits = [False, True] * 5
    # plt.errorbar(x, y, yerr=yerr, uplims=upperlimits, lolims=lowerlimits,
    #              label='subsets of uplims and lolims')

    plt.legend(loc='lower right')
    plt.show()
