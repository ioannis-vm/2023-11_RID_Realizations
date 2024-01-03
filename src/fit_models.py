"""
Fit all models to the data and store the parameters.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
from src.models import Model_0_P58
from src.models import Model_1_Weibull
from src.models import Model_1v0_Weibull
from src.models import Model_1v1_Weibull


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

    # # limit to given hazard level
    # case_df = case_df.xs('2', level=0).dropna()

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
    model.censoring_limit = 0.002
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


    ###############################
    # Weibull with two parameters #
    ###############################

    model = Model_1v0_Weibull()
    model.parameters = (0.03, 1.30)

    # Confirms:
    # - that the pdf looks the same when plotted in Mathematica
    rid = np.linspace(1e-8, 0.08, 1000)
    pid = np.linspace(1e-8, 0.08, 1000)
    # pdf_values = model.evaluate_pdf(rid, pid)
    # fig, ax = plt.subplots()
    # ax.plot(rid, pdf_values)
    # plt.show()

    # Confirms:
    # - that the empirical CDF of the generated samples matches the
    #   theoretical CDF
    rid_samples = model.generate_rid_samples(pid)
    # cdf_val = model.evaluate_cdf(rid_samples, pid)
    # fig, ax = plt.subplots()
    # sns.ecdfplot(rid_samples, ax=ax)
    # ax.scatter(rid_samples, cdf_val, color='C1', s=1)
    # plt.show()

    # Fit another model to get back the parameters
    other_model = Model_1v0_Weibull()
    other_model.add_data(pid, rid_samples)
    other_model.fit(method='mle') # Can change to `quantiles` for experimentation.
    print(other_model.parameters)
    fig, ax = plt.subplots()
    model.plot_model(ax, rolling=False, training=False, model=True, model_color='C0')
    other_model.plot_model(ax, rolling=False, training=True, model=True, model_color='C1')
    ax.legend(
        [
         Line2D([0], [0], color='C0', lw=1),
         Line2D([0], [0], color='C1', lw=1),
        ],
        ['Assigned', 'Retrieved']
    )
    plt.show()

    #########################################
    # Weibull with two parameters, censored #
    #########################################

    model = Model_1v0_Weibull()
    model.parameters = (0.03, 1.30)

    rid = np.linspace(1e-8, 0.08, 1000)
    pid = np.linspace(1e-8, 0.08, 1000)

    rid_samples = model.generate_rid_samples(pid)

    other_model = Model_1v0_Weibull()
    other_model.add_data(pid, rid_samples)
    other_model.censoring_limit = 0.002 # Modify this. Works up to 0.07! wow.
    other_model.fit(method='mle')
    print(other_model.parameters)
    fig, ax = plt.subplots()
    model.plot_model(ax, rolling=False, training=False, model=True, model_color='C0')
    other_model.plot_model(ax, rolling=False, training=True, model=True, model_color='C1')
    ax.axvline(x=other_model.censoring_limit, linestyle='dashed', color='black')
    ax.legend(
        [
         Line2D([0], [0], color='C0', lw=1),
         Line2D([0], [0], color='C1', lw=1),
        ],
        ['Assigned', 'Retrieved']
    )
    plt.show()

    ##########################################
    # Weibull with linear lambda function #
    ##########################################

    model = Model_1v1_Weibull()
    model.parameters = (0.30, 1.30) # c_lamda_slope, c_kapa

    rid = np.linspace(1e-8, 0.08, 1000)
    pid = np.linspace(1e-8, 0.08, 1000)

    rid_samples = model.generate_rid_samples(pid)
    
    other_model = Model_1v1_Weibull()
    other_model.add_data(pid, rid_samples)
    other_model.censoring_limit = None
    other_model.fit(method='mle')
    print(other_model.parameters)
    plt.close()
    fig, ax = plt.subplots()
    model.plot_model(ax, rolling=False, training=False, model=True, model_color='C0')
    other_model.plot_model(ax, rolling=False, training=False, model=True, model_color='C1')
    if other_model.censoring_limit:
        ax.axvline(x=other_model.censoring_limit, linestyle='dashed', color='black')
    ax.legend(
        [
         Line2D([0], [0], color='C0', lw=1),
         Line2D([0], [0], color='C1', lw=1),
        ],
        ['Assigned', 'Retrieved']
    )
    plt.show()

    ##########################################
    # Weibull with piecewise lambda function #
    ##########################################

    model = Model_1_Weibull()
    model.parameters = (0.01, 0.30, 1.30) # c_lamda_slope, c_kapa

    rid = np.linspace(1e-8, 0.08, 1000)
    pid = np.linspace(1e-8, 0.08, 1000)

    rid_samples = model.generate_rid_samples(pid)
    
    other_model = Model_1_Weibull()
    other_model.add_data(pid, rid_samples)
    other_model.censoring_limit = 0.002
    other_model.fit(method='mle')
    print(other_model.parameters)
    plt.close()
    fig, ax = plt.subplots()
    model.plot_model(ax, rolling=False, training=False, model=True, model_color='C0')
    other_model.plot_model(ax, rolling=False, training=False, model=True, model_color='C1')
    ax.axvline(x=other_model.censoring_limit, linestyle='dashed', color='black')
    ax.legend(
        [
         Line2D([0], [0], color='C0', lw=1),
         Line2D([0], [0], color='C1', lw=1),
        ],
        ['Assigned', 'Retrieved']
    )
    plt.show()

    # Fit multiple times, get distribution for the parameters
    
    def get_params():
        model = Model_1_Weibull()
        model.parameters = (0.01, 0.30, 1.30) # c_lamda_slope, c_kapa

        rid = np.linspace(1e-8, 0.08, 1000)
        pid = np.linspace(1e-8, 0.08, 1000)

        rid_samples = model.generate_rid_samples(pid)

        other_model = Model_1_Weibull()
        other_model.add_data(pid, rid_samples)
        other_model.censoring_limit = 0.002
        other_model.fit(method='mle')
        return other_model.parameters

    params = []
    from tqdm import trange
    for _ in trange(100):
        params.append(get_params())

    res = np.concatenate(params)
    res = res.reshape(-1, 3)

    fig, axs = plt.subplots(3, 1)
    sns.ecdfplot(res[:, 0], ax=axs[0])
    axs[0].axvline(x=0.01)
    axs[0].set(xlim=(0.00, 0.02))
    sns.ecdfplot(res[:, 1], ax=axs[1])
    axs[1].axvline(x=0.30)
    axs[1].set(xlim=(0.00, 0.60))
    sns.ecdfplot(res[:, 2], ax=axs[2])
    axs[2].axvline(x=1.30)
    axs[2].set(xlim=(0.00, 2.60))
    plt.show()

if __name__ == '__main__':

    main()
