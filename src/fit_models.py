"""
Fit all models to the data and store the parameters.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.models import Model_0_P58
from src.models import Model_1_Weibull


def main():
    """
    This is not the right place for this method: will be moved.
    """

    from src.handle_data import load_dataset
    from src.handle_data import remove_collapse
    from src.handle_data import only_drifts

    the_case = ("smrf", "3", "ii", "1", "1")
    # the_case = ("scbf", "9", "ii", "3", "1")
    # the_case = ("brbf", "9", "ii", "1", "1")

    df = only_drifts(remove_collapse(load_dataset()[0]))
    case_df = df[the_case].dropna()

    # limit to given hazard level
    case_df = case_df.xs('2', level=0).dropna()

    rid_vals = case_df.dropna()["RID"].to_numpy().reshape(-1)
    pid_vals = case_df.dropna()["PID"].to_numpy().reshape(-1)

    # FEMA P-58

    model = Model_0_P58()
    model.add_data(pid_vals, rid_vals)
    model.fit(0.007, 0.90)      # values hand-picked for ("smrf", "9", "ii", "1", "1")

    fig, ax = plt.subplots()
    model.plot_model(ax)
    ax.set(xlim=(-0.005, 0.08), ylim=(-0.005, 0.08))
    ax.fill_between((0.002, 0.015), -1, 1, alpha=0.30, color='black')
    plt.show()


    # Weibull, MLE

    model = Model_1_Weibull()
    model.add_data(pid_vals, rid_vals)
    model.fit(method='mle')

    fig, ax = plt.subplots()
    model.plot_model(ax)
    ax.set(xlim=(-0.005, 0.025), ylim=(-0.005, 0.06))
    ax.fill_between((0.002, 0.015), -1, 1, alpha=0.30, color='black')
    plt.show()

    # plot a slice
    pid_min = 0.01
    pid_max = 0.015
    fig, ax = plt.subplots()
    model.plot_slice(ax, pid_min, pid_max, 0.002)
    plt.show()

    # Weibull, quantile regression

    model = Model_1_Weibull()
    model.add_data(pid_vals, rid_vals)
    model.fit(method='quantiles')

    fig, ax = plt.subplots()
    model.plot_model(ax)
    ax.set(xlim=(-0.005, 0.025), ylim=(-0.005, 0.06))
    ax.fill_between((0.002, 0.015), -1, 1, alpha=0.30, color='black')
    plt.show()

    # plot a slice
    pid_min = 0.01
    pid_max = 0.015
    fig, ax = plt.subplots()
    model.plot_slice(ax, pid_min, pid_max, 0.002)
    plt.show()


    # Weibull, MLE on generated samples

    model = Model_1_Weibull()
    model.params = np.array((0.0045 , 0.3, 2.50))

    pid_samples = np.random.choice(pid_vals, size=1000, replace=True)
    # pid_samples = np.random.uniform(0.0, 0.06, size=1000)
    rid_samples = model.generate_rid_samples(pid_samples)

    model = Model_1_Weibull()
    model.add_data(pid_samples, rid_samples)
    model.fit(method='quantiles')
    print(model.params)
    # np.array((0.0045 , 0.3, 2.50))

    model = Model_1_Weibull()
    model.add_data(pid_samples, rid_samples)
    model.fit(method='mle')
    print(model.params)
    # np.array((0.0045 , 0.3, 2.50))


    fig, ax = plt.subplots()
    model.plot_model(ax)
    ax.set(xlim=(-0.005, 0.025), ylim=(-0.005, 0.06))
    ax.fill_between((0.002, 0.015), -1, 1, alpha=0.30, color='black')
    plt.show()

    # plot a slice
    pid_min = 0.01
    pid_max = 0.015
    fig, ax = plt.subplots()
    model.plot_slice(ax, pid_min, pid_max, 0.002)
    
    plt.show()
