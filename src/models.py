"""
RID Project
RID|PID models
"""

import numpy as np
from scipy.special import erfc
from scipy.special import erfcinv
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import seaborn as sns


np.set_printoptions(formatter={'float': '{:0.5f}'.format})

class Model:
    """
    Base Model class.
    """

    def __init__(self):

        self.raw_pid = None
        self.raw_rid = None

        self.sim_pid = None
        self.sim_rid = None

        self.censoring_limit = None
        self.parameters = None
        self.fit_status = False
        self.fit_meta = None

        self.rolling_pid = None
        self.rolling_rid_50 = None
        self.rolling_rid_20 = None
        self.rolling_rid_80 = None

    def add_data(self, raw_pid, raw_rid):

        self.raw_pid = raw_pid
        self.raw_rid = raw_rid


    def calculate_rolling_quantiles(self):

        # calculate rolling empirical RID|PID quantiles
        idsort = np.argsort(self.raw_pid)
        num_vals = len(self.raw_pid)
        assert len(self.raw_rid) == num_vals
        rid_vals_sorted = self.raw_rid[idsort]
        pid_vals_sorted = self.raw_pid[idsort]
        group_size = int(len(self.raw_rid) * 0.10)
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

        self.rolling_pid = rolling_pid
        self.rolling_rid_50 = rolling_rid_50
        self.rolling_rid_20 = rolling_rid_20
        self.rolling_rid_80 = rolling_rid_80


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

    def generate_rid_samples(self, pid_samples):
        """
        Generates RID samples given PID samples, from the conditional
        distribution
        """
        raise NotImplementedError("Subclasses should implement this.")

    def plot_data(self, ax=None, scatter_kwargs=None):
        """
        Add a scatter plot of the raw data to a matplotlib axis, or
        show it if one is not given.
        """

        if scatter_kwargs == None:
            scatter_kwargs = {
                's': 5.0,
                'color': 'black',
                'alpha': 0.1
            }
        
        if ax is None:
            _, ax = plt.subplots()
        ax.scatter(
            self.raw_rid,
            self.raw_pid,
            **scatter_kwargs
        )
        if ax is None:
            plt.show()

    def plot_model(self, ax, rolling=True, training=True, model=True, model_color='C0'):
        """
        Plot the data in a scatter plot,
        superimpose their empirical quantiles,
        and the quantiles resulting from the fitted model.
        """

        if self.fit_status == 'False':
            self.fit()

        rid_vals = self.raw_rid
        pid_vals = self.raw_pid

        if training:
            self.plot_data(ax)
        
        if rolling:
            self.calculate_rolling_quantiles()

            ax.plot(self.rolling_rid_50, self.rolling_pid, 'k')
            ax.plot(self.rolling_rid_20, self.rolling_pid, 'k', linestyle='dashed')
            ax.plot(self.rolling_rid_80, self.rolling_pid, 'k', linestyle='dashed')

        if model:
            # model_pid = np.linspace(0.00, self.rolling_pid[-1], 1000)
            model_pid = np.linspace(0.00, 0.08, 1000)
            model_rid_50 = self.evaluate_inverse_cdf(0.50, model_pid)
            model_rid_20 = self.evaluate_inverse_cdf(0.20, model_pid)
            model_rid_80 = self.evaluate_inverse_cdf(0.80, model_pid)

            ax.plot(model_rid_50, model_pid, model_color)
            ax.plot(model_rid_20, model_pid, model_color, linestyle='dashed')
            ax.plot(model_rid_80, model_pid, model_color, linestyle='dashed')

    def plot_slice(self, ax, pid_min, pid_max, censoring_limit=None):

        mask = (self.raw_pid > pid_min) & (self.raw_pid < pid_max)
        vals = self.raw_rid[mask]
        midpoint = np.mean((pid_min, pid_max))
        sns.ecdfplot(vals, color=f'C0', ax=ax)
        if censoring_limit:
            ax.axvline(x=censoring_limit, color='black')
        x = np.linspace(0.00, 0.05, 1000)
        y = self.evaluate_cdf(x, np.array((midpoint,)))
        ax.plot(x, y, color='C1')


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

    def fit(self, delta_y, beta, **kwargs):
        """
        The P-58 model requires the user to specify the parameters
        directly.
        """
        self.parameters = (delta_y, beta)

    def evaluate_inverse_cdf(self, quantile, pid_vals):
        """
        Evaluate the inverse of the conditional RID|PID CDF.
        """
        delta_y, beta = self.parameters
        delta_val = self.delta_fnc(pid_vals, delta_y)
        return delta_val * np.exp(-np.sqrt(2.00) * beta * erfcinv(2.00 * quantile))

    def evaluate_cdf(self, rid, pid):
        """
        Evaluate the conditional RID|PID CDF.
        """
        delta_y, beta = self.parameters
        delta_val = self.delta_fnc(pid, delta_y)
        return 0.50 * erfc(-((np.log(rid / delta_val))) / (np.sqrt(2.0) * beta))


class Model_1v0_Weibull(Model):
    """
    Weibull model with two parameters
    """

    def evaluate_pdf(self, rid, pid, censoring_limit=None):

        c_lamda, c_kapa = self.parameters

        pdf_val =  (
            np.exp(-((rid / c_lamda) ** c_kapa))
            * c_kapa
            * (rid / c_lamda) ** (c_kapa - 1.00)
        ) / c_lamda
        if censoring_limit:
            censored_range_mass = self.evaluate_cdf(
                np.full(len(pid), censoring_limit), pid)
            mask = rid <= censoring_limit
            pdf_val[mask] = censored_range_mass[mask]
        return pdf_val

    def evaluate_cdf(self, rid, pid):
        c_lamda, c_kapa = self.parameters
        return 1.00 - np.exp(-((rid / c_lamda) ** c_kapa))

    def evaluate_inverse_cdf(self, q, pid):
        c_lamda, c_kapa = self.parameters
        return np.full(len(pid), c_lamda) * (-np.log(1.00 - q))**(1.00 / c_kapa)

    def get_mle_objective(self, parameters):

        # update the parameters
        self.parameters = parameters

        density = self.evaluate_pdf(
            self.raw_rid, self.raw_pid, self.censoring_limit
        )
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

    def fit(self, method='quantile', **kwargs):

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
            tol=1e-8
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
        pdf_val = (
            np.exp(-((rid / lamda_val) ** c_kapa))
            * c_kapa
            * (rid / lamda_val) ** (c_kapa - 1.00)
        ) / lamda_val
        if censoring_limit:
            censored_range_mass = self.evaluate_cdf(np.full(len(pid), censoring_limit), pid)
            mask = rid <= censoring_limit
            pdf_val[mask] = censored_range_mass[mask]
        return pdf_val

    def evaluate_cdf(self, rid, pid):
        _, c_kapa = self.parameters
        lamda_val = self.lamda_fnc(pid)
        return 1.00 - np.exp(-((rid / lamda_val) ** c_kapa))

    def evaluate_inverse_cdf(self, q, pid):
        c_lamda_slope, c_kapa = self.parameters
        lamda_val = self.lamda_fnc(pid)
        return lamda_val * (-np.log(1.00 - q))**(1.00 / c_kapa)

    def get_mle_objective(self, parameters):

        # update the parameters
        self.parameters = parameters
        density = self.evaluate_pdf(
            self.raw_rid, self.raw_pid, self.censoring_limit
        )
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

    def fit(self, method='quantile', **kwargs):

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
            tol=1e-8
        )
        self.fit_meta = result
        assert result.success, "Minimization failed."
        self.parameters = result.x

    def generate_rid_samples(self, pid_samples):

        u = np.random.uniform(0.00, 1.00, len(pid_samples))
        rid_samples = self.evaluate_inverse_cdf(u, pid_samples)
        return rid_samples


class Model_1_Weibull(Model):
    """
    Weibull model
    """

    def lamda_fnc(self, pid):
        c_pid_0, c_lamda_slope, _ = self.parameters
        c_lamda_0 = 1.0e-8
        lamda = np.ones_like(pid) * c_lamda_0
        mask = pid >= c_pid_0
        lamda[mask] = (pid[mask] - c_pid_0) * c_lamda_slope + c_lamda_0
        return lamda

    def evaluate_pdf(self, rid, pid, censoring_limit=None):
        _, _, c_kapa = self.parameters
        lamda_val = self.lamda_fnc(pid)
        pdf_val = (
            np.exp(-((rid / lamda_val) ** c_kapa))
            * c_kapa
            * (rid / lamda_val) ** (c_kapa - 1.00)
        ) / lamda_val
        if censoring_limit:
            censored_range_mass = self.evaluate_cdf(np.full(len(pid), censoring_limit), pid)
            mask = rid <= censoring_limit
            pdf_val[mask] = censored_range_mass[mask]
        return pdf_val

    def evaluate_cdf(self, rid, pid):
        _, _, c_kapa = self.parameters
        lamda_val = self.lamda_fnc(pid)
        return 1.00 - np.exp(-((rid / lamda_val) ** c_kapa))

    def evaluate_inverse_cdf(self, q, pid):
        c_pid_0, c_lamda_slope, c_kapa = self.parameters
        lamda_val = self.lamda_fnc(pid)
        return lamda_val * (-np.log(1.00 - q))**(1.00 / c_kapa)

    def get_mle_objective(self, parameters):

        # update the parameters
        self.parameters = parameters
        density = self.evaluate_pdf(
            self.raw_rid, self.raw_pid, self.censoring_limit
        )
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

    def fit(self, method='quantile', **kwargs):

        # Initial values
        c_pid_0 = 0.008
        c_lamda_slope = 0.30
        c_kapa = 1.30

        self.parameters = (c_pid_0, c_lamda_slope, c_kapa)

        if method == 'quantiles':
            use_method = self.get_quantile_objective
        elif method == 'mle':
            use_method = self.get_mle_objective

        result = minimize(
            use_method,
            [c_pid_0, c_lamda_slope, c_kapa],
            method="Nelder-Mead",
            options={"maxiter": 10000},
            tol=1e-8
        )
        self.fit_meta = result
        assert result.success, "Minimization failed."
        self.parameters = result.x

    def generate_rid_samples(self, pid_samples):

        u = np.random.uniform(0.00, 1.00, len(pid_samples))
        rid_samples = self.evaluate_inverse_cdf(u, pid_samples)
        return rid_samples
