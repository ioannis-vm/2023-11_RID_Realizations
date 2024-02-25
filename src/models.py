"""
RID Project
RID|PID models
"""

# pylint:disable=no-name-in-module


import numpy as np
import scipy as sp
from scipy.interpolate import interp1d
from scipy.special import erfc
from scipy.special import erfcinv
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import seaborn as sns


np.set_printoptions(formatter={'float': '{:0.5f}'.format})


def lognormal_fragility_weight(rid, censoring_limit=None, delta=0.01, beta=0.60):
    """
    Determine MLE weights based on the RID value, considering the
    lognormal residual drift fragility curve for which the generated
    RIDs will be the input. delta is the median and beta the
    dispersion of the fragility curve.  We use the shape of its
    density function, scaled to a maximum weight, and shifted in the Y
    axis to have a weight of 1 far away from the median.
    """
    max_scaling = 5.00

    # scaling_factors = (max_scaling - 1.00) * (
    #     delta
    #     / (np.exp((beta**4 + np.log(rid / delta) ** 2) / (2 * beta**2)) * rid)
    # ) + 1.00

    exponent = (beta**4 + np.log(rid / delta) ** 2) / (2 * beta**2)
    scaling_factors = np.empty_like(rid)
    # 700 is a rough threshold to avoid overflow in exp
    #   In this case the function evaluates to 1.00
    scaling_factors[exponent > 700.00] = 1.00
    scaling_factors[exponent <= 700.00] = (max_scaling - 1.00) * (
        delta / (np.exp(exponent[exponent <= 700.00]) * rid[exponent <= 700.00])
    ) + 1.00

    # # also account for the marginal distribution of the RIDs we want
    # # each potential RID value to have the same baseline weight
    # # regardless of the number of data points in that neighborhood

    # if censoring_limit:
    #     bins = np.concatenate(
    #         (
    #             np.array((0.00, censoring_limit)),
    #             np.linspace(censoring_limit, np.max(rid), 24),
    #         )
    #     )
    # else:
    #     bins = (np.linspace(0.00, np.max(censoring_limit), 24),)
    # hist_bin_vals = np.histogram(rid, bins)[0]
    # hist_fun = interp1d(
    #     bins[:-1], hist_bin_vals, kind='previous', fill_value='extrapolate'
    # )
    # hist_vals = hist_fun(rid)

    # scaling_factors = scaling_factors / ((hist_vals - 1.00)/50.00 + 1.00)

    return scaling_factors


class Model:
    """
    Base Model class.
    """

    def __init__(self):
        self.raw_pid = None
        self.raw_rid = None

        self.uniform_sample = None
        self.sim_pid = None
        self.sim_rid = None

        self.censoring_limit = None
        self.parameters = None
        self.parameter_names = None
        self.parameter_bounds = None
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

    def fit(self, *args, **kwargs):
        """
        Obtain the parameters by fitting the model to the data
        """
        raise NotImplementedError("Subclasses should implement this.")

    def evaluate_inverse_cdf(self, quantile, pid):
        """
        Evaluate the inverse of the conditional RID|PID CDF.
        """
        raise NotImplementedError("Subclasses should implement this.")

    def evaluate_pdf(self, rid, pid, censoring_limit=None):
        """
        Evaluate the conditional RID|PID PDF.
        """
        raise NotImplementedError("Subclasses should implement this.")

    def evaluate_cdf(self, rid, pid):
        """
        Evaluate the conditional RID|PID CDF.
        """
        raise NotImplementedError("Subclasses should implement this.")

    def get_mle_objective(self, parameters):
        # update the parameters
        self.parameters = parameters
        density = self.evaluate_pdf(self.raw_rid, self.raw_pid, self.censoring_limit)
        negloglikelihood = -np.sum(np.log(density))
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

    def generate_rid_samples(self, pid_samples):
        if self.uniform_sample is None:
            self.uniform_sample = np.random.uniform(0.00, 1.00, len(pid_samples))
        rid_samples = self.evaluate_inverse_cdf(self.uniform_sample, pid_samples)

        self.sim_pid = pid_samples
        self.sim_rid = rid_samples

        return rid_samples

    def plot_data(self, ax=None, scatter_kwargs=None):
        """
        Add a scatter plot of the raw data to a matplotlib axis, or
        show it if one is not given.
        """

        if scatter_kwargs is None:
            scatter_kwargs = {
                's': 5.0,
                'facecolor': 'none',
                'edgecolor': 'black',
                'alpha': 0.2,
            }

        if ax is None:
            _, ax = plt.subplots()
        ax.scatter(self.raw_rid, self.raw_pid, **scatter_kwargs)
        if ax is None:
            plt.show()

    def plot_model(
        self, ax, rolling=True, training=True, model=True, model_color='C0'
    ):
        """
        Plot the data in a scatter plot,
        superimpose their empirical quantiles,
        and the quantiles resulting from the fitted model.
        """

        if self.fit_status == 'False':
            self.fit()

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
        sns.ecdfplot(vals, color='C0', ax=ax)
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

    def fit(self, *args, delta_y=0.00, beta=0.00, **kwargs):
        """
        The P-58 model requires the user to specify the parameters
        directly.
        """
        self.parameters = (delta_y, beta)

    def evaluate_inverse_cdf(self, quantile, pid):
        """
        Evaluate the inverse of the conditional RID|PID CDF.
        """
        delta_y, beta = self.parameters
        delta_val = self.delta_fnc(pid, delta_y)
        return delta_val * np.exp(-np.sqrt(2.00) * beta * erfcinv(2.00 * quantile))

    def evaluate_cdf(self, rid, pid):
        """
        Evaluate the conditional RID|PID CDF.
        """
        delta_y, beta = self.parameters
        delta_val = self.delta_fnc(pid, delta_y)
        return 0.50 * erfc(-((np.log(rid / delta_val))) / (np.sqrt(2.0) * beta))


class BilinearModel(Model):
    """
    One parameter constant, the other varies in a bilinear fashion.
    """

    def bilinear_fnc(self, pid):
        theta_1_a, c_lamda_slope, _ = self.parameters
        c_lamda_0 = 1.0e-8
        lamda = np.ones_like(pid) * c_lamda_0
        mask = pid >= theta_1_a
        lamda[mask] = (pid[mask] - theta_1_a) * c_lamda_slope + c_lamda_0
        return lamda

    def evaluate_inverse_cdf(self, quantile, pid):
        """
        Evaluate the inverse of the conditional RID|PID CDF.
        """
        raise NotImplementedError("Subclasses should implement this.")

    def evaluate_pdf(self, rid, pid, censoring_limit=None):
        """
        Evaluate the conditional RID|PID PDF.
        """
        raise NotImplementedError("Subclasses should implement this.")

    def evaluate_cdf(self, rid, pid):
        """
        Evaluate the conditional RID|PID CDF.
        """
        raise NotImplementedError("Subclasses should implement this.")

    def fit(self, *args, method='mle', **kwargs):
        # Initial values

        if method == 'quantiles':
            use_method = self.get_quantile_objective
        elif method == 'mle':
            use_method = self.get_mle_objective

        result = minimize(
            use_method,
            self.parameters,
            bounds=self.parameter_bounds,
            method="Nelder-Mead",
            options={"maxiter": 10000},
            tol=1e-6,
        )
        self.fit_meta = result
        assert result.success, "Minimization failed."
        self.parameters = result.x


class Model_1_Weibull(BilinearModel):
    """
    Weibull model
    """

    def __init__(self):
        super().__init__()
        # initial parameters
        self.parameters = np.array((0.008, 0.30, 1.30))
        # parameter names
        self.parameter_names = ('pid_0', 'lambda_slope', 'kappa')
        # bounds
        self.parameter_bounds = ((0.00, 0.02), (0.00, 1.00), (0.80, 4.00))

    def evaluate_pdf(self, rid, pid, censoring_limit=None):
        _, _, theta_3 = self.parameters
        bilinear_fnc_val = self.bilinear_fnc(pid)
        pdf_val = sp.stats.weibull_min.pdf(rid, theta_3, 0.00, bilinear_fnc_val)
        pdf_val[pdf_val < 1e-6] = 1e-6
        if censoring_limit:
            censored_range_mass = self.evaluate_cdf(
                np.full(len(pid), censoring_limit), pid
            )
            mask = rid <= censoring_limit
            pdf_val[mask] = censored_range_mass[mask]
        return pdf_val

    def evaluate_cdf(self, rid, pid):
        _, _, theta_3 = self.parameters
        bilinear_fnc_val = self.bilinear_fnc(pid)
        return sp.stats.weibull_min.cdf(rid, theta_3, 0.00, bilinear_fnc_val)

    def evaluate_inverse_cdf(self, quantile, pid):
        _, _, theta_3 = self.parameters
        bilinear_fnc_val = self.bilinear_fnc(pid)
        return sp.stats.weibull_min.ppf(quantile, theta_3, 0.00, bilinear_fnc_val)


class Model_2_Gamma(BilinearModel):
    """
    Gamma model
    """

    def __init__(self):
        super().__init__()
        # initial parameters
        self.parameters = np.array((0.008, 0.30, 1.30))
        # parameter names
        self.parameter_names = ('pid_0', 'lambda_slope', 'kappa')
        # bounds
        self.parameter_bounds = ((0.00, 0.02), (0.00, 1.00), (0.80, 4.00))

    def evaluate_pdf(self, rid, pid, censoring_limit=None):
        _, _, theta_3 = self.parameters
        bilinear_fnc_val = self.bilinear_fnc(pid)
        pdf_val = sp.stats.gamma.pdf(rid, theta_3, 0.00, bilinear_fnc_val)
        pdf_val[pdf_val < 1e-6] = 1e-6
        if censoring_limit:
            censored_range_mass = self.evaluate_cdf(
                np.full(len(pid), censoring_limit), pid
            )
            mask = rid <= censoring_limit
            pdf_val[mask] = censored_range_mass[mask]
        return pdf_val

    def evaluate_cdf(self, rid, pid):
        _, _, theta_3 = self.parameters
        bilinear_fnc_val = self.bilinear_fnc(pid)
        return sp.stats.gamma.cdf(rid, theta_3, 0.00, bilinear_fnc_val)

    def evaluate_inverse_cdf(self, quantile, pid):
        _, _, theta_3 = self.parameters
        bilinear_fnc_val = self.bilinear_fnc(pid)
        return sp.stats.gamma.ppf(quantile, theta_3, 0.00, bilinear_fnc_val)


class Model_3_Beta(BilinearModel):
    """
    Beta model
    """

    def __init__(self):
        super().__init__()
        # initial parameters
        self.parameters = np.array((0.004, 100.00, 500.00))
        # parameter names
        self.parameter_names = ('pid_0', 'alpha_slope', 'beta')
        # bounds
        self.parameter_bounds = None

    def evaluate_pdf(self, rid, pid, censoring_limit=None):
        _, _, c_beta = self.parameters
        c_alpha = self.bilinear_fnc(pid)
        pdf_val = sp.stats.beta.pdf(rid, c_alpha, c_beta)
        pdf_val[pdf_val < 1e-6] = 1e-6
        if censoring_limit:
            censored_range_mass = self.evaluate_cdf(
                np.full(len(pid), censoring_limit), pid
            )
            mask = rid <= censoring_limit
            pdf_val[mask] = censored_range_mass[mask]
        return pdf_val

    def evaluate_cdf(self, rid, pid):
        _, _, c_beta = self.parameters
        c_alpha = self.bilinear_fnc(pid)
        return sp.stats.beta.cdf(rid, c_alpha, c_beta)

    def evaluate_inverse_cdf(self, quantile, pid):
        _, _, c_beta = self.parameters
        c_alpha = self.bilinear_fnc(pid)
        return sp.stats.beta.ppf(quantile, c_alpha, c_beta)
