"""
Fit all models to the data and store the parameters.
"""

import matplotlib.pyplot as plt
from src.models import Model_0_P58
from src.models import Model_1_Weibull
from src.models import Model_2_Gamma
from src.models import Model_3_Beta
from src.handle_data import load_dataset
from src.handle_data import remove_collapse
from src.handle_data import only_drifts


def main():
    the_case = ("scbf", "9", "ii", "3", "2")

    df = only_drifts(remove_collapse(load_dataset('data/edp_extra.parquet')[0]))
    case_df = df[the_case].dropna()

    # # limit to given hazard level
    # case_df = case_df.xs('2', level=0).dropna()

    rid_vals = case_df.dropna()["RID"].to_numpy().reshape(-1)
    pid_vals = case_df.dropna()["PID"].to_numpy().reshape(-1)

    # # >>>>>>>>>>>>>>>>>>>>>>>

    # import numpy as np
    # from scipy.interpolate import interp1d

    # vals, bins = np.histogram(rid_vals)
    # vals = np.append(vals, 1).astype(float)
    # ifun = interp1d(bins, vals, kind='previous', fill_value='extrapolate')
    # x = np.linspace(0.00, 0.05, 1000)
    # y = ifun(x)

    # # <<<<<<<<<<<<<<<<<<<<<<<

    # FEMA P-58
    model = Model_0_P58()
    model.add_data(pid_vals, rid_vals)
    model.fit(
        delta_y=0.01, beta=0.60
    )  # values hand-picked for ("smrf", "9", "ii", "1", "1")

    _, ax = plt.subplots()
    model.plot_model(ax)
    ax.set(xlim=(-0.005, 0.08), ylim=(-0.005, 0.08))
    plt.show()

    # Weibull, MLE

    model = Model_1_Weibull()
    model.add_data(pid_vals, rid_vals)
    model.censoring_limit = 0.0005
    model.fit(method='mle')

    _, ax = plt.subplots()
    model.plot_model(ax)
    ax.set(xlim=(-0.005, 0.025), ylim=(-0.005, 0.06))
    plt.show()

    # plot a slice
    pid_min = 0.025
    pid_max = 0.035
    _, ax = plt.subplots()
    model.plot_slice(ax, pid_min, pid_max, 0.002)
    plt.show()

    # Weibull, quantile regression

    model = Model_1_Weibull()
    model.add_data(pid_vals, rid_vals)
    model.fit(method='quantiles')

    _, ax = plt.subplots()
    model.plot_model(ax)
    ax.set(xlim=(-0.005, 0.025), ylim=(-0.005, 0.06))
    plt.show()

    # plot a slice
    pid_min = 0.025
    pid_max = 0.035
    _, ax = plt.subplots()
    model.plot_slice(ax, pid_min, pid_max, 0.002)
    plt.show()

    # Gamma, MLE

    model = Model_2_Gamma()
    model.add_data(pid_vals, rid_vals)
    model.censoring_limit = 0.0005
    model.fit(method='mle')

    _, ax = plt.subplots()
    model.plot_model(ax)
    ax.set(xlim=(-0.005, 0.025), ylim=(-0.005, 0.06))
    plt.show()

    # plot a slice
    pid_min = 0.025
    pid_max = 0.035
    _, ax = plt.subplots()
    model.plot_slice(ax, pid_min, pid_max, 0.002)
    plt.show()

    # Beta, MLE

    model = Model_3_Beta()
    model.add_data(pid_vals, rid_vals)
    model.censoring_limit = 0.0005
    model.fit(method='mle')

    _, ax = plt.subplots()
    model.plot_model(ax)
    ax.set(xlim=(-0.005, 0.025), ylim=(-0.005, 0.06))
    plt.show()

    # plot a slice
    pid_min = 0.025
    pid_max = 0.035
    _, ax = plt.subplots()
    model.plot_slice(ax, pid_min, pid_max, 0.002)
    plt.show()


if __name__ == '__main__':
    main()

