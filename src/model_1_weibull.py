import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import gaussian_kde
import seaborn as sns
import matplotlib.pyplot as plt
from src.handle_data import load_dataset
from src.handle_data import remove_collapse
from src.handle_data import only_drifts


def lamda_fnc(pid, c_lamda_init, c_pid_0, c_lamda_slope):
    lamda = np.ones_like(pid) * c_lamda_init
    mask = pid >= c_pid_0
    lamda[mask] = (pid[mask] - c_pid_0) * c_lamda_slope + c_lamda_init
    return lamda


def evaluate_density(rid, pid, c_lamda_init, c_pid_0, c_lamda_slope, c_kapa):
    lamda_val = lamda_fnc(pid, c_lamda_init, c_pid_0, c_lamda_slope)
    return (
        np.exp(-((rid / lamda_val) ** c_kapa))
        * c_kapa
        * (rid / lamda_val) ** (c_kapa - 1.00)
    ) / lamda_val


def evaluate_joint_density(rid, pid, c_lamda_init, c_pid_0, c_lamda_slope, c_kapa):
    conditional_density = evaluate_density(rid, pid, c_lamda_init, c_pid_0, c_lamda_slope, c_kapa)
    kde = gaussian_kde(pid[:, 0])
    pid_marginal = kde(pid[:, 0])
    return conditional_density * pid_marginal


def fit_distribution(rid_vals, pid_vals):
    def get_objective(params):
        c_lamda, c_pid_0, c_lamda_slope, c_kapa = params
        density = evaluate_density(
            rid_vals, pid_vals, c_lamda, c_pid_0, c_lamda_slope, c_kapa
        )
        density = density[density > 1e-4]  # ignore outliers
        loglikelihood = np.sum(np.log(density))
        return -loglikelihood

    c_lamda = 1e-6
    c_pid_0 = 0.011
    c_lamda_slope = 0.39
    c_kapa = 1.30

    result = minimize(
        get_objective,
        [c_lamda, c_pid_0, c_lamda_slope, c_kapa],
        method="nelder-mead",
        options={"maxiter": 2000},
    )
    assert result.success

    c_lamda, c_pid_0, c_lamda_slope, c_kapa = result.x

    return c_lamda, c_pid_0, c_lamda_slope, c_kapa


def plot_data(case_df, given_ax=None):

    if given_ax is None:
        _, ax = plt.subplots()
    else:
        ax = given_ax
    sns.scatterplot(
        data=case_df,
        x="RID",
        y="PID",
        hue=case_df.index.get_level_values("hz"),
        ax=ax
    )
    # ax.set(xlim=(-0.005, 0.04), ylim=(-0.005, 0.08))
    # ax.grid(which="both", linewidth=0.30)
    if given_ax is None:
        plt.show()



instance = ("smrf", "3", "ii", "1", "1")

df = only_drifts(remove_collapse(load_dataset()[0]))
case_df = df[instance]
rid_vals = case_df.dropna()["RID"].to_numpy().reshape(-1)
pid_vals = case_df.dropna()["PID"].to_numpy().reshape(-1)
params = fit_distribution(rid_vals, pid_vals)

# # kde plot of the pid marginal density
# plt.hist(pid_vals, bins=25, density=True)
# kde = gaussian_kde(pid_vals)
# kde(pid_vals)
# pid_vals.sort()
# plt.plot(pid_vals, kde(pid_vals))
# plt.show()

plot_data(case_df)

# plotting the CDF of RID on a slice of the data (for a small range of
# PID)

pid_min = 0.02
pid_max = 0.03

fig, ax = plt.subplots()
mask = (case_df['PID'] > pid_min) & (case_df['PID'] < pid_max)
vals = case_df[mask]['RID']
midpoint = np.mean((pid_min, pid_max))
sns.ecdfplot(vals, color=f'C0', ax=ax)
x = np.linspace(0.00, 0.05, 1000)
lmdval = lamda_fnc(np.array((midpoint,)), *params[0:3])
y = (
    1.00 - np.exp(-(x/lmdval)**params[3])
)
ax.plot(x, y, color=f'C0')
plt.show()



# contour plot of the joint density

fig, ax = plt.subplots()
plot_data(case_df, ax)
rids = np.linspace(0.00, 0.08, 100)
pids = np.linspace(0.00, 0.08, 100)
X, Y = np.meshgrid(rids, pids)
Z = evaluate_joint_density(X, Y, *params)
plt.contour(X, Y, Z, levels=100, linewidths=0.5, cmap='viridis')
plt.show()

