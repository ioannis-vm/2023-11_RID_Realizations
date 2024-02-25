"""
Check if the parametric distribution matches that of the data by
looking for quantile uniformity
"""

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import seaborn as sns
from src import models
from src.handle_data import load_dataset
from src.handle_data import remove_collapse
from src.handle_data import only_drifts


def main():
    the_case = ("scbf", "9", "ii", "3")  # we combine the two directions

    df = only_drifts(remove_collapse(load_dataset('data/edp_extra.parquet')[0]))
    case_df = df[the_case].dropna().stack(level=0)

    analysis_rid_vals = case_df.dropna()["RID"].to_numpy().reshape(-1)
    analysis_pid_vals = case_df.dropna()["PID"].to_numpy().reshape(-1)

    model = models.Model_1_Weibull()
    model.add_data(analysis_pid_vals, analysis_rid_vals)
    model.censoring_limit = 0.0025
    model.fit(method='mle')

    # Scatter plot, model's theoretical quantiles
    _, ax = plt.subplots()
    model.plot_model(ax)
    ax.set(xlim=(-0.005, 0.025), ylim=(-0.005, 0.06))
    plt.show()

    # plot ECDFs for a slice
    pid_min = 0.025
    pid_max = 0.035
    _, ax = plt.subplots()
    model.plot_slice(ax, pid_min, pid_max, 0.002)
    plt.show()

    # sample data from the model
    simulated_pids = np.random.choice(analysis_pid_vals, size=2000, replace=True)
    simulated_rids = model.generate_rid_samples(simulated_pids)

    # get quantiles
    quants = model.evaluate_cdf(simulated_rids, simulated_pids)

    # they are uniformly distributred in [0, 1]
    # note: that is the case regardless of the marginal distribution
    #       of pids
    fig, ax = plt.subplots()
    ax.hist(quants, bins=25, density=True)
    plt.show()

    # Now make the same plot with the original data
    # rid = analysis_rid_vals
    # pid = analysis_pid_vals
    rid = analysis_rid_vals[analysis_rid_vals > model.censoring_limit]
    pid = analysis_pid_vals[analysis_rid_vals > model.censoring_limit]
    quants = model.evaluate_cdf(rid, pid)

    fig, ax = plt.subplots()
    ax.hist(
        quants,
        bins=25,
        range=(0.00, 1.00),
        density=True,
        edgecolor='black',
        facecolor='lightblue',
        zorder=10,
    )
    ax.fill_between([0.0, 1.0], 0.00, 1.00, zorder=-1, alpha=0.20)
    plt.show()

    # Scatter plot, model's theoretical quantiles, points colored
    # based on bin frequency
    bins = np.linspace(0.00, 1.00, num=26)
    hist_bin_vals = np.histogram(quants, bins, density='True')[0]

    hist_fun = interp1d(
        bins[:-1], hist_bin_vals, kind='previous', fill_value='extrapolate'
    )
    hist_vals = hist_fun(quants)

    _, ax = plt.subplots()
    model.plot_model(ax, training=False, rolling=False)
    sns.scatterplot(
        data=pd.DataFrame({'x': rid, 'y': pid, 's': hist_vals}),
        x='x',
        y='y',
        hue='s',
        ax=ax,
    )
    ax.set(xlim=(-0.005, 0.025), ylim=(-0.005, 0.06))
    plt.show()


if __name__ == '__main__':
    main()
