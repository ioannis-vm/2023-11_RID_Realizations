"""
Experimental fitting with simplified models
"""

import numpy as np
import scipy as sp
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
from tqdm import trange
from src.models import Model
from src.models import Model_1_Weibull


# pylint: disable=missing-function-docstring, arguments-differ


class Model_1v0_Weibull(Model):
    """
    Weibull model with two parameters
    """

    def evaluate_pdf(self, rid, pid, censoring_limit=None):
        c_lamda, c_kapa = self.parameters

        pdf_val = sp.stats.weibull_min.pdf(rid, c_kapa, 0.00, c_lamda)
        # pdf_val =  (
        #     np.exp(-((rid / c_lamda) ** c_kapa))
        #     * c_kapa
        #     * (rid / c_lamda) ** (c_kapa - 1.00)
        # ) / c_lamda
        if censoring_limit:
            censored_range_mass = self.evaluate_cdf(
                np.full(len(pid), censoring_limit), pid
            )
            mask = rid <= censoring_limit
            pdf_val[mask] = censored_range_mass[mask]
        return pdf_val

    def evaluate_cdf(self, rid, pid):
        c_lamda, c_kapa = self.parameters
        # return 1.00 - np.exp(-((rid / c_lamda) ** c_kapa))
        return sp.stats.weibull_min.cdf(rid, c_kapa, 0.00, c_lamda)

    def evaluate_inverse_cdf(self, quantile, pid):
        c_lamda, c_kapa = self.parameters
        lamda = np.full(len(pid), c_lamda)
        kapa = np.full(len(pid), c_kapa)
        # return np.full(len(pid), c_lamda) * (-np.log(1.00 - q))**(1.00 / c_kapa)
        return sp.stats.weibull_min.ppf(quantile, kapa, 0.00, lamda)

    def get_mle_objective(self, parameters):
        # update the parameters
        self.parameters = parameters

        density = self.evaluate_pdf(self.raw_rid, self.raw_pid, self.censoring_limit)
        weights = np.ones_like(density)
        # mask = (self.raw_pid > 0.002) & (self.raw_pid < 0.015)
        # weights[mask] = 1.0     # no weights
        negloglikelihood = -np.sum(np.log(density) * weights)
        return negloglikelihood

    def get_quantile_objective(self, parameters):
        # update the parameters
        self.parameters = parameters

        # calculate the model's RID|PID quantiles
        if self.rolling_pid is None:
            self.calculate_rolling_quantiles()
        model_pid = self.rolling_pid
        model_rid_50 = self.evaluate_inverse_cdf(0.50, model_pid)
        model_rid_20 = self.evaluate_inverse_cdf(0.20, model_pid)
        model_rid_80 = self.evaluate_inverse_cdf(0.80, model_pid)

        loss = (
            (self.rolling_rid_50 - model_rid_50).T
            @ (self.rolling_rid_50 - model_rid_50)
            + (self.rolling_rid_20 - model_rid_20).T
            @ (self.rolling_rid_20 - model_rid_20)
            + (self.rolling_rid_80 - model_rid_80).T
            @ (self.rolling_rid_80 - model_rid_80)
        )

        return loss

    def fit(self, method='quantile'):
        # Initial values
        c_lamda = 0.30
        c_kapa = 1.30

        self.parameters = (c_lamda, c_kapa)

        if method == 'quantiles':
            use_method = self.get_quantile_objective
        elif method == 'mle':
            use_method = self.get_mle_objective

        result = minimize(
            use_method,
            [c_lamda, c_kapa],
            method="Nelder-Mead",
            options={"maxiter": 10000},
            tol=1e-8,
        )
        self.fit_meta = result
        assert result.success, "Minimization failed."
        self.parameters = result.x

    def generate_rid_samples(self, pid_samples):
        u = np.random.uniform(0.00, 1.00, len(pid_samples))
        rid_samples = self.evaluate_inverse_cdf(u, pid_samples)
        return rid_samples


class Model_1v1_Weibull(Model):
    """
    Weibull model with linear lambda function
    """

    def lamda_fnc(self, pid):
        c_lamda_slope, _ = self.parameters
        lamda = pid * c_lamda_slope
        return lamda

    def evaluate_pdf(self, rid, pid, censoring_limit=None):
        _, c_kapa = self.parameters
        lamda_val = self.lamda_fnc(pid)
        # pdf_val = (
        #     np.exp(-((rid / lamda_val) ** c_kapa))
        #     * c_kapa
        #     * (rid / lamda_val) ** (c_kapa - 1.00)
        # ) / lamda_val
        pdf_val = sp.stats.weibull_min.pdf(rid, c_kapa, 0.00, lamda_val)
        if censoring_limit:
            censored_range_mass = self.evaluate_cdf(
                np.full(len(pid), censoring_limit), pid
            )
            mask = rid <= censoring_limit
            pdf_val[mask] = censored_range_mass[mask]
        return pdf_val

    def evaluate_cdf(self, rid, pid):
        _, c_kapa = self.parameters
        lamda_val = self.lamda_fnc(pid)
        # return 1.00 - np.exp(-((rid / lamda_val) ** c_kapa))
        return sp.stats.weibull_min.cdf(rid, c_kapa, 0.00, lamda_val)

    def evaluate_inverse_cdf(self, quantile, pid):
        _, c_kapa = self.parameters
        lamda_val = self.lamda_fnc(pid)
        # return lamda_val * (-np.log(1.00 - q))**(1.00 / c_kapa)
        return sp.stats.weibull_min.ppf(quantile, c_kapa, 0.00, lamda_val)

    def get_mle_objective(self, parameters):
        # update the parameters
        self.parameters = parameters
        density = self.evaluate_pdf(self.raw_rid, self.raw_pid, self.censoring_limit)
        weights = np.ones_like(density)
        # mask = (self.raw_pid > 0.002) & (self.raw_pid < 0.015)
        # weights[mask] = 1.0     # no weights
        negloglikelihood = -np.sum(np.log(density) * weights)
        return negloglikelihood

    def get_quantile_objective(self, parameters):
        # update the parameters
        self.parameters = parameters

        # calculate the model's RID|PID quantiles
        if self.rolling_pid is None:
            self.calculate_rolling_quantiles()
        model_pid = self.rolling_pid
        model_rid_50 = self.evaluate_inverse_cdf(0.50, model_pid)
        model_rid_20 = self.evaluate_inverse_cdf(0.20, model_pid)
        model_rid_80 = self.evaluate_inverse_cdf(0.80, model_pid)

        loss = (
            (self.rolling_rid_50 - model_rid_50).T
            @ (self.rolling_rid_50 - model_rid_50)
            + (self.rolling_rid_20 - model_rid_20).T
            @ (self.rolling_rid_20 - model_rid_20)
            + (self.rolling_rid_80 - model_rid_80).T
            @ (self.rolling_rid_80 - model_rid_80)
        )

        return loss

    def fit(self, method='quantile'):
        # Initial values
        c_lamda_slope = 0.30
        c_kapa = 1.30

        self.parameters = (c_lamda_slope, c_kapa)

        if method == 'quantiles':
            use_method = self.get_quantile_objective
        elif method == 'mle':
            use_method = self.get_mle_objective

        result = minimize(
            use_method,
            [c_lamda_slope, c_kapa],
            method="Nelder-Mead",
            options={"maxiter": 10000},
            tol=1e-8,
        )
        self.fit_meta = result
        assert result.success, "Minimization failed."
        self.parameters = result.x

    def generate_rid_samples(self, pid_samples):
        u = np.random.uniform(0.00, 1.00, len(pid_samples))
        rid_samples = self.evaluate_inverse_cdf(u, pid_samples)
        return rid_samples


def main():

    ###############################
    # Weibull with two parameters #
    ###############################

    model = Model_1v0_Weibull()
    model.parameters = (0.03, 1.30)

    # Confirms:
    # - that the pdf looks the same when plotted in Mathematica
    pid = np.linspace(1e-8, 0.08, 1000)
    # rid = pid
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
    other_model.fit(method='mle')  # Can change to `quantiles` for experimentation.
    print(other_model.parameters)
    _, ax = plt.subplots()
    model.plot_model(ax, rolling=False, training=False, model=True, model_color='C0')
    other_model.plot_model(
        ax, rolling=False, training=True, model=True, model_color='C1'
    )
    ax.legend(
        [
            Line2D([0], [0], color='C0', lw=1),
            Line2D([0], [0], color='C1', lw=1),
        ],
        ['Assigned', 'Retrieved'],
    )
    plt.show()

    #########################################
    # Weibull with two parameters, censored #
    #########################################

    model = Model_1v0_Weibull()
    model.parameters = (0.03, 1.30)

    pid = np.linspace(1e-8, 0.08, 1000)
    rid_samples = model.generate_rid_samples(pid)

    other_model = Model_1v0_Weibull()
    other_model.add_data(pid, rid_samples)
    other_model.censoring_limit = 0.002  # Modify this. Works up to 0.07! wow.
    other_model.fit(method='mle')
    print(other_model.parameters)
    _, ax = plt.subplots()
    model.plot_model(ax, rolling=False, training=False, model=True, model_color='C0')
    other_model.plot_model(
        ax, rolling=False, training=True, model=True, model_color='C1'
    )
    ax.axvline(x=other_model.censoring_limit, linestyle='dashed', color='black')
    ax.legend(
        [
            Line2D([0], [0], color='C0', lw=1),
            Line2D([0], [0], color='C1', lw=1),
        ],
        ['Assigned', 'Retrieved'],
    )
    plt.show()

    ##########################################
    # Weibull with linear lambda function #
    ##########################################

    model = Model_1v1_Weibull()
    model.parameters = (0.30, 1.30)  # c_lamda_slope, c_kapa

    pid = np.linspace(1e-8, 0.08, 1000)
    rid_samples = model.generate_rid_samples(pid)

    other_model = Model_1v1_Weibull()
    other_model.add_data(pid, rid_samples)
    other_model.censoring_limit = None
    other_model.fit(method='mle')
    print(other_model.parameters)
    plt.close()
    _, ax = plt.subplots()
    model.plot_model(ax, rolling=False, training=False, model=True, model_color='C0')
    other_model.plot_model(
        ax, rolling=False, training=False, model=True, model_color='C1'
    )
    if other_model.censoring_limit:
        ax.axvline(x=other_model.censoring_limit, linestyle='dashed', color='black')
    ax.legend(
        [
            Line2D([0], [0], color='C0', lw=1),
            Line2D([0], [0], color='C1', lw=1),
        ],
        ['Assigned', 'Retrieved'],
    )
    plt.show()

    ##########################################
    # Weibull with piecewise lambda function #
    ##########################################

    model = Model_1_Weibull()
    model.parameters = (0.01, 0.30, 1.30)  # c_lamda_slope, c_kapa

    pid = np.linspace(1e-8, 0.08, 1000)
    rid_samples = model.generate_rid_samples(pid)

    other_model = Model_1_Weibull()
    other_model.add_data(pid, rid_samples)
    other_model.censoring_limit = 0.002
    other_model.fit(method='mle')
    print(other_model.parameters)
    plt.close()
    _, ax = plt.subplots()
    model.plot_model(ax, rolling=False, training=False, model=True, model_color='C0')
    other_model.plot_model(
        ax, rolling=False, training=False, model=True, model_color='C1'
    )
    ax.axvline(x=other_model.censoring_limit, linestyle='dashed', color='black')
    ax.legend(
        [
            Line2D([0], [0], color='C0', lw=1),
            Line2D([0], [0], color='C1', lw=1),
        ],
        ['Assigned', 'Retrieved'],
    )
    plt.show()

    # Fit multiple times, get distribution for the parameters

    def get_params():
        model = Model_1_Weibull()
        model.parameters = (0.01, 0.30, 1.30)  # c_lamda_slope, c_kapa

        pid = np.linspace(1e-8, 0.08, 1000)
        rid_samples = model.generate_rid_samples(pid)

        other_model = Model_1_Weibull()
        other_model.add_data(pid, rid_samples)
        other_model.censoring_limit = 0.002
        other_model.fit(method='mle')
        return other_model.parameters

    params = []

    for _ in trange(100):
        params.append(get_params())

    res = np.concatenate(params)
    res = res.reshape(-1, 3)

    _, axs = plt.subplots(3, 1)
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
