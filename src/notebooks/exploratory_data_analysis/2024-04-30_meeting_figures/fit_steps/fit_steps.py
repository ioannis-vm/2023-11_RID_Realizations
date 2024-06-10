"""
Figures to demonstrate the steps involved in fitting a model.

"""

# Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src import models

# import seaborn as sns

# plotting parameters
xmin, xmax = -0.00099, 0.0221  # rid
ymin, ymax = -0.00499, 0.0699  # pid
# sns.set_style("whitegrid")


df = pd.read_parquet('data/edp_extended_cms.parquet')
df.index = df.index.reorder_levels(
    ['system', 'stories', 'rc', 'dir', 'edp', 'hz', 'gm', 'loc']
)
df = df.sort_index()


def get_pid_rid_pairs(system, stories, rc, direction):
    """
    Retrieves PID-RID pairs for the given case.

    Parameters
    ----------
    system: str
        Any of {smrf, scbf, brbf}
    stories: str
        Any of {3, 6, 9}
    rc: str
        Any of {ii, iv}
    direction: str
        Any of {x, y}

    Returns
    -------
    pd.DataFrame
        Dataframe with PID-RID pairs, with unstable collapse cases
        removed (transient drift exceeding 10%).

    """
    # subset based on above parameters
    df_archetype = df.loc[(system, stories, rc, direction), :]
    # get PID and RID
    pid = df_archetype.loc['PID', :]['value']
    pid = pid.groupby(by=['hz', 'gm']).max()
    rid = df_archetype.loc['RID', :]['value']
    rid = rid.groupby(by=['hz', 'gm']).max()

    # remove cases where PID > 0.10 (unstable collapse)
    idx_keep = pid[pid < 0.10].index
    pid = pid.loc[idx_keep]
    rid = rid.loc[idx_keep]

    pairs = pd.concat((pid, rid), keys=['PID', 'RID'], axis=1)
    pairs = pairs.reset_index()
    for thing in ('hz', 'gm'):
        pairs[thing] = pairs[thing].astype(float)
    return pairs


system = 'smrf'
stories = '3'
rc = 'ii'
yield_drift = {
    'smrf': 1.30,
    'scbf': 0.45,
    'brbf': 0.50,
}


data_x = get_pid_rid_pairs(system, stories, rc, 'x')
data_y = get_pid_rid_pairs(system, stories, rc, 'y')
data = pd.concat((data_x, data_y))


fig, ax = plt.subplots(figsize=(3, 3))

# ax.fill_between(
#     np.array((xmin, xmax)),
#     slice_loc - slice_width,
#     slice_loc + slice_width,
#     alpha=0.20,
# )

ax.scatter(
    data['RID'].values, data['PID'].values, color='darkgray', marker='.', s=0.1
)
model = models.Model_1_Weibull()
model.add_data(data['PID'].values, data['RID'].values)
model.censoring_limit = 0.0025
model.fit(method='mle')

model.calculate_rolling_quantiles()

ax.plot(
    model.rolling_rid_50,
    model.rolling_pid,
    'k',
    linewidth=2.5,
    label='Empirical',
)
ax.plot(model.rolling_rid_20, model.rolling_pid, 'k', linestyle='dashed')
ax.plot(model.rolling_rid_80, model.rolling_pid, 'k', linestyle='dashed')

model_pid = np.linspace(0.00, 0.06, 1000)
model_rid_50 = model.evaluate_inverse_cdf(0.50, model_pid)
model_rid_20 = model.evaluate_inverse_cdf(0.20, model_pid)
model_rid_80 = model.evaluate_inverse_cdf(0.80, model_pid)

ax.plot(
    model_rid_50,
    model_pid,
    'C0',
    linewidth=2.5,
    label='Conditional Weibull',
    zorder=10,
)
ax.plot(model_rid_20, model_pid, 'C0', linestyle='dashed', zorder=10, linewidth=2)
ax.plot(model_rid_80, model_pid, 'C0', linestyle='dashed', zorder=10, linewidth=2)

ax.axvline(x=0.0025, color='black', linestyle='dashed')

# model.plot_model(ax)
ax.set(
    xlim=(xmin, xmax),
    ylim=(ymin, ymax),
    xlabel='RID',
    ylabel='PID',
)
# ax.legend(frameon=False, loc='lower right')
ax.grid(which='both')
fig.tight_layout()
# plt.show()
plt.savefig('/tmp/fig.png', dpi=600)

# data_sub = data[data['PID'] < slice_loc + slice_width]
# data_sub = data_sub[data_sub['PID'] > slice_loc - slice_width]
# fig, ax = plt.subplots(figsize=(3, 3))
# sns.ecdfplot(data_sub['RID'], ax=ax, label='Empirical', color='black')
# ax.set(
#     xlim=(xmin, xmax),
#     xlabel='Largest RID across floors',
# )
# rid_proposed = model.generate_rid_samples(np.full(1000, slice_loc))
# sns.ecdfplot(rid_proposed, ax=ax, label='Conditional Weibull', color='C0')
# rid_fema = fema_model.generate_rid_samples(np.full(1000, slice_loc))
# sns.ecdfplot(rid_fema, ax=ax, label='FEMA P-58', color='C1')
# ax.legend(frameon=False, loc='lower right')
# ax.grid(which='both')
# fig.tight_layout()
# # plt.show()
# plt.savefig(f'/tmp/{system}_{stories}_{rc}_cdf.png', dpi=600)
