"""
RID Project
RID|PID models
"""

import numpy as np
import pandas as pd
from scipy.special import erfc
from scipy.special import erfcinv
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import seaborn as sns


class Model:
    """
    Base Model class.
    """

    def __init__(self, raw_pid, raw_rid):
        self.raw_pid = raw_pid
        self.raw_rid = raw_rid
        self.params = None
        self.fit_status = False
        self.fit_meta = None
    
    def plot_data(self, ax=None):
        """
        Add a scatter plot of the raw data to a matplotlib axis, or
        show it if one is not given.
        """
        if ax is None:
            _, ax = plt.subplots()
        ax.scatter(
            self.raw_rid,
            self.raw_pid,
            s=5.0,
            color='black'
        )
        if ax is None:
            plt.show()

    def fit(self):
        """
        Obtain the parameters by fitting the model to the data
        """
        raise NotImplementedError("Subclasses should implement this.")

    def evaluate_inverse_cdf(self):
        """
        Evaluate the inverse of the conditional RID|PID CDF.
        """
        raise NotImplementedError("Subclasses should implement this.")

    def evaluate_cdf(self):
        """
        Evaluate the conditional RID|PID CDF.
        """
        raise NotImplementedError("Subclasses should implement this.")

    def plot_model(self, ax):
        """
        Plot the data in a scatter plot,
        superimpose their empirical quantiles,
        and the quantiles resulting from the fitted model.
        """

        if self.fit_status == 'False':
            self.fit()

        rid_vals = self.raw_rid
        pid_vals = self.raw_pid

        # calculate rolling empirical RID|PID quantiles
        idsort = np.argsort(pid_vals)
        num_vals = len(pid_vals)
        assert len(rid_vals) == num_vals
        rid_vals_sorted = rid_vals[idsort]
        pid_vals_sorted = pid_vals[idsort]
        group_size = int(len(rid_vals) * 0.10)
        rolling_pid = np.array(
            [
                np.mean(pid_vals_sorted[i : i + group_size])
                for i in range(num_vals - group_size + 1)
            ]
        )
        rolling_rid_50 = np.array(
            [
                np.quantile(rid_vals_sorted[i : i + group_size], 0.50)
                for i in range(num_vals - group_size + 1)
            ]
        )
        rolling_rid_20 = np.array(
            [
                np.quantile(rid_vals_sorted[i : i + group_size], 0.20)
                for i in range(num_vals - group_size + 1)
            ]
        )
        rolling_rid_80 = np.array(
            [
                np.quantile(rid_vals_sorted[i : i + group_size], 0.80)
                for i in range(num_vals - group_size + 1)
            ]
        )

        # calculate the model's RID|PID quantiles
        # model_pid = np.linspace(0.00, 0.01, 1000)
        model_pid = rolling_pid
        model_rid_50 = self.evaluate_inverse_cdf(0.50, model_pid)
        model_rid_20 = self.evaluate_inverse_cdf(0.20, model_pid)
        model_rid_80 = self.evaluate_inverse_cdf(0.80, model_pid)

        self.plot_data(ax)
        ax.plot(rolling_rid_50, rolling_pid, 'k')
        ax.plot(rolling_rid_20, rolling_pid, 'k', linestyle='dashed')
        ax.plot(rolling_rid_80, rolling_pid, 'k', linestyle='dashed')
        ax.plot(model_rid_50, model_pid, 'C0')
        ax.plot(model_rid_20, model_pid, 'C0', linestyle='dashed')
        ax.plot(model_rid_80, model_pid, 'C0', linestyle='dashed')
        ax.grid(which='both', linewidth=0.30)

    def plot_slice(self, ax, pid_min, pid_max):

        mask = (self.raw_pid > pid_min) & (self.raw_pid < pid_max)
        vals = self.raw_rid[mask]
        midpoint = np.mean((pid_min, pid_max))
        sns.ecdfplot(vals, color=f'C0', ax=ax)
        x = np.linspace(0.00, 0.05, 1000)
        y = self.evaluate_cdf(x, np.array((midpoint,)))
        ax.plot(x, y, color='C0')


class Model_0_P58(Model):
    """
    FEMA P-58 model.
    """

    def delta_fnc(self, pid, delta_y):
        delta = np.zeros_like(pid)
        mask = (delta_y <= pid) & (pid < 4.0 * delta_y)
        delta[mask] = 0.30 * (pid[mask] - delta_y)
        mask = 4.0 * delta_y <= pid
        delta[mask] = pid[mask] - 3.00 * delta_y
        return delta

    def fit(self, delta_y, beta):
        """
        The P-58 model requires the user to specify the parameters
        directly.
        """
        self.params = (delta_y, beta)

    def evaluate_inverse_cdf(self, quantile, pid_vals):
        """
        Evaluate the inverse of the conditional RID|PID CDF.
        """
        delta_y, beta = self.params
        delta_val = self.delta_fnc(pid_vals, delta_y)
        return delta_val * np.exp(-np.sqrt(2.00) * beta * erfcinv(2.00 * quantile))

    def evaluate_cdf(self, rid, pid):
        """
        Evaluate the conditional RID|PID CDF.
        """
        delta_y, beta = self.params
        delta_val = self.delta_fnc(pid, delta_y)
        return 0.50 * erfc(-((np.log(rid / delta_val))) / (np.sqrt(2.0) * beta))


class Model_1_Weibull(Model):
    """
    Weibull model
    """

    def lamda_fnc(self, pid, c_lamda_0, c_pid_0, c_lamda_slope):
        lamda = np.ones_like(pid) * c_lamda_0
        mask = pid >= c_pid_0
        lamda[mask] = (pid[mask] - c_pid_0) * c_lamda_slope + c_lamda_0
        return lamda

    def evaluate_pdf(self, rid, pid, c_lamda_0, c_pid_0, c_lamda_slope, c_kapa):
        lamda_val = self.lamda_fnc(pid, c_lamda_0, c_pid_0, c_lamda_slope)
        return (
            np.exp(-((rid / lamda_val) ** c_kapa))
            * c_kapa
            * (rid / lamda_val) ** (c_kapa - 1.00)
        ) / lamda_val

    def evaluate_cdf(self, rid, pid):
        c_lamda_0, c_pid_0, c_lamda_slope, c_kapa = self.params
        lamda_val = self.lamda_fnc(pid, c_lamda_0, c_pid_0, c_lamda_slope)
        return 1.00 - np.exp(-((rid / lamda_val) ** c_kapa))

    def evaluate_inverse_cdf(self, q, pid):
        c_lamda_0, c_pid_0, c_lamda_slope, c_kapa = self.params
        lamda_val = self.lamda_fnc(pid, c_lamda_0, c_pid_0, c_lamda_slope)
        return lamda_val * (-np.log(1.00 - q))**(1.00 / c_kapa)

    def get_objective(self, params):
        c_lamda, c_pid_0, c_lamda_slope, c_kapa = params
        density = self.evaluate_pdf(
            self.raw_rid, self.raw_pid, c_lamda, c_pid_0, c_lamda_slope, c_kapa
        )

        # zero_idx = np.argwhere(density < 1e-6).reshape(-1)
        # nozr_idx = np.argwhere(density >= 1e-6).reshape(-1)
        # plt.scatter(rid_vals[zero_idx], pid_vals[zero_idx], color='red')
        # plt.scatter(rid_vals[nozr_idx], pid_vals[nozr_idx], color='blue', s=2)
        # plt.show()

        # density += 1.00               # regularization
        density[density < 1e-6] = 1e-6  # outlier removal
        negloglikelihood = -np.sum(np.log(density))
        return negloglikelihood

    def fit(self):
        c_lamda = 1e-12
        c_pid_0 = 0.008
        c_lamda_slope = 0.30
        c_kapa = 1.30

        # self.get_objective((c_lamda, c_pid_0, c_lamda_slope, c_kapa))

        result = minimize(
            self.get_objective,
            [c_lamda, c_pid_0, c_lamda_slope, c_kapa],
            method="Nelder-Mead",
            options={"maxiter": 20000},
            tol=1e-8
        )
        self.fit_meta = result
        assert result.success, "Minimization failed."
        self.params = result.x



from src.handle_data import load_dataset
from src.handle_data import remove_collapse
from src.handle_data import only_drifts


def main():

    the_case = ("smrf", "9", "ii", "1", "1")

    df = only_drifts(remove_collapse(load_dataset()[0]))
    case_df = df[the_case]

    rid_vals = case_df.dropna()["RID"].to_numpy().reshape(-1)
    pid_vals = case_df.dropna()["PID"].to_numpy().reshape(-1)

    model = Model_1_Weibull(pid_vals, rid_vals)
    # model = Model_0_P58(pid_vals, rid_vals)

    model.fit()
    model.fit(0.007, 0.90)
    fig, ax = plt.subplots()
    model.plot_model(ax)
    ax.set(xlim=(-0.005, 0.08), ylim=(-0.005, 0.08))
    ax.fill_between((0.002, 0.015), -1, 1, alpha=0.30, color='black')
    plt.show()

    # manual fit...
    #               c_lamda,    c_pid_0, c_lamda_slope, c_kapa
    model.params = (1.0e-12,  0.007,   5.0e-01,       1.0)
    fig, ax = plt.subplots()
    model.plot_model(ax)
    ax.set(xlim=(-0.005, 0.08), ylim=(-0.005, 0.08))
    ax.fill_between((0.002, 0.015), -1, 1, alpha=0.30, color='black')
    plt.show()

    # model.get_objective(model.params)

    # slice
    pid_min = 0.007
    pid_max = 0.012
    fig, ax = plt.subplots()
    model.plot_slice(ax, pid_min, pid_max)
    plt.show()
