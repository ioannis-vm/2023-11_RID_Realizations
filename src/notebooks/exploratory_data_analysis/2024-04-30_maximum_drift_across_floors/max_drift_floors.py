# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%

# %% [markdown]
"""
# Examining the viability of a maximum drift model

`Sat Apr 27 09:53:47 AM PDT 2024`

With this notebook we try to see if we could work with the maximum
drift observed across all stories instead of having to simulate the
RID-PID relationship story by story. This way, we would be able to
generate realizations only for the maximum observed residual drift
across floors—which ultimately drives the excessive drift
consequences—given the maximum observed peak transient drift across
floors.
"""

# %% [markdown]
"""
### Setup
"""

# %%
# Imports
from pathlib import Path
import os
from itertools import product
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Change directory to project's root, if needed
# (not needed when re-evaluating)
if Path(os.getcwd()).name != '2023-11_RID_Realizations':
    os.chdir('../../')

# from temp.max_drift.methods import test

# %%
# plotting parameters
xmin, xmax = -0.005, 0.04  # rid
ymin, ymax = -0.005, 0.08  # pid
sns.set_style("whitegrid")


# %%
df = pd.read_parquet('data/edp_extended_cs.parquet')
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


# %%
for system, stories, rc in product(
    ('smrf', 'scbf', 'brbf'), ('3', '6', '9'), ('ii', 'iv')
):
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    for i, direction in enumerate(('x', 'y')):
        sns.scatterplot(
            data=get_pid_rid_pairs(system, stories, rc, direction),
            x='RID',
            y='PID',
            hue='hz',
            ax=axs[i],
        )
        sns.despine(ax=axs[i])
        axs[i].set(
            xlim=(xmin, xmax),
            ylim=(ymin, ymax),
            title=' '.join((system, stories, rc, direction)),
        )
    plt.show()

# %% [markdown]
"""
**Comment on the results**: It's clear that a pattern exists.
We would like to know whether the maximum PID—maximum RID pairs
correspond to the same story, or if there is a large number of cases
where they don't.
To find out, we construct a bar chart with the number of cases where
the pair corresponds to the same story versus otherwise.
"""


# %%
def num_story_difference(system, stories, rc, direction, censoring=False):
    """
    Max pid across stories is compared against max RID across
    stories. But they might not occur at the same story. This function
    checks where they occur and calculates the difference. E.g. if max
    PID is story 1 and max RID is story 2, it returns 1. If the max's
    occur at the same story, returns 0. The actual return type is a
    pd.Series for all hazard level - ground motion combinations
    corresponding to the given inputs.

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
    censoring: bool
        If True, only consider results where RID is higher than 0.25%.

    Returns
    -------
    pd.DataFrame
        in `story difference`, it contains the number of stories
        between where the maximum PID was observed versus where the
        maximum RID was observed.
        in `large drift` it contains a boolean flag indicating that
        PID was greater than 0.25%.

    """
    # subset based on above parameters
    df_archetype = df.loc[(system, stories, rc, direction), :]
    # get PID and RID
    pid = df_archetype.loc['PID', :]['value']
    pid_idxmax = pid.groupby(by=['hz', 'gm']).idxmax()
    pid = pid.groupby(by=['hz', 'gm']).max()
    rid = df_archetype.loc['RID', :]['value']
    rid_idxmax = rid.groupby(by=['hz', 'gm']).idxmax()
    rid = rid.groupby(by=['hz', 'gm']).max()

    # remove cases where PID > 0.10 (unstable collapse)
    idx_keep = pid[pid < 0.10].index
    pid_idxmax = pid_idxmax.loc[idx_keep]
    rid_idxmax = rid_idxmax.loc[idx_keep]
    rid = rid.loc[idx_keep]
    pid = pid.loc[idx_keep]

    if censoring:
        idx_keep = rid[rid > 0.025].index
        pid_idxmax = pid_idxmax.loc[idx_keep]
        rid_idxmax = rid_idxmax.loc[idx_keep]

    pairs = pd.concat((pid_idxmax, rid_idxmax), keys=['PID', 'RID'], axis=1)

    for column in pairs.columns:
        pairs[column] = pairs[column].apply(lambda x: int(x[-1]))

    pairs['story difference'] = (pairs['RID'] - pairs['PID']).abs()
    pairs['large drift'] = rid > 0.025

    return pairs


# %%
for rc, stories, system in product(
    ('ii', 'iv'),
    ('3', '6', '9'),
    ('smrf', 'scbf', 'brbf'),
):
    fig, axs = plt.subplots(1, 2, figsize=(6, 2))
    for i, censoring in enumerate((False, True)):
        sns.countplot(
            data=num_story_difference(system, stories, rc, 'x', censoring),
            x='story difference',
            ax=axs[i],
        )
        sns.despine(ax=axs[i])
        axs[i].set(
            title=' '.join((system, stories, rc, 'x', str(censoring))),
        )
    plt.show()

# %% [markdown]
"""
Note: The plots are only for the x direction, but the y direction is
    very similar.
**Comment on the results**:
- Considering the results as a whole, for taller archetypes usually
    the maximum PID occurs at a different floor than the maximum PID.
- The same applies to a lesser extent for RC IV, but the fact that we
    have fewer large PID-RID pairs for those might have an impact.
- Focusing specifically on above-censoring RID results, it's much more
    common for the RID-PID pair to be coming from the same story.
"""
