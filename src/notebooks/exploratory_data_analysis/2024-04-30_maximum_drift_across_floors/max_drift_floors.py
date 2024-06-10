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
# pylint: disable=wrong-import-position
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
    os.chdir('../../../../')


# %%
# plotting parameters
xmin, xmax = -0.004, 0.04  # rid
ymin, ymax = -0.004, 0.08  # pid
sns.set_style("whitegrid")


# %%
df = pd.read_parquet('data/edp_extended_old/edp_extended_cs.parquet')
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
        removed (transient drift exceeding 10%). It also contains the
        story where the maximum PID or RID was observed.

    """

    # subset based on above parameters
    df_archetype = df.loc[(system, stories, rc, direction), :]
    # get PID and RID
    pid = df_archetype.loc['PID', :]['value']
    pid_story = pid.groupby(by=['hz', 'gm']).idxmax().apply(lambda x: int(x[-1]))
    pid = pid.groupby(by=['hz', 'gm']).max()
    rid = df_archetype.loc['RID', :]['value']
    rid_story = rid.groupby(by=['hz', 'gm']).idxmax().apply(lambda x: int(x[-1]))
    rid = rid.groupby(by=['hz', 'gm']).max()

    # remove cases where PID > 0.10 (unstable collapse)
    idx_keep = pid[pid < 0.10].index
    pid = pid.loc[idx_keep]
    pid_story = pid_story.loc[idx_keep]
    rid = rid.loc[idx_keep]
    rid_story = rid_story.loc[idx_keep]

    pairs = pd.concat(
        (pid, rid, pid_story, rid_story),
        keys=['PID', 'RID', 'PIDStory', 'RIDStory'],
        axis=1,
    )
    pairs = pairs.reset_index()
    pairs['PID'] = pairs['PID'].astype(float)
    pairs['RID'] = pairs['RID'].astype(float)
    pairs['PIDStory'] = pairs['PIDStory'].astype(int)
    pairs['RIDStory'] = pairs['RIDStory'].astype(int)

    # Partitioning used for plotting
    pairs['StoryDiff'] = (pairs['PIDStory'] - pairs['RIDStory']).abs()
    pairs['StoryDiffText'] = ''
    pairs.loc[pairs['StoryDiff'] == 0, 'StoryDiffText'] = 'Same story'
    pairs.loc[pairs['StoryDiff'] == 1, 'StoryDiffText'] = '1--2 stories'
    pairs.loc[pairs['StoryDiff'] == 2, 'StoryDiffText'] = '1--2 stories'
    pairs.loc[pairs['StoryDiff'] > 2, 'StoryDiffText'] = '>2 stories'
    for thing in ('PIDStory', 'RIDStory'):
        pairs[f'{thing}Text'] = 'Intermediate stories'
        if stories == '3':
            pairs.loc[pairs[thing] <= 2, f'{thing}Text'] = 'First 2 stories'
        else:
            pairs.loc[pairs[thing] <= 3, f'{thing}Text'] = 'First 3 stories'
        pairs.loc[pairs[thing] == int(stories), f'{thing}Text'] = 'Last story'

    return pairs


# %% [markdown]
"""
### Are maximum RIDs observed at the same story as that of the PIDs?
"""

# %%
for system, stories, rc in product(
    ('smrf', 'scbf', 'brbf'), ('3', '6', '9'), ('ii', 'iv')
):
    plt.close()
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    for i, direction in enumerate(('x', 'y')):
        sns.scatterplot(
            data=get_pid_rid_pairs(system, stories, rc, direction),
            x='RID',
            y='PID',
            hue='StoryDiffText',
            ax=axs[i],
            palette={
                'Same story': "#7B9F35",
                '1--2 stories': "#226666",
                '>2 stories': "#AA3939",
            },
            s=10,
        )
        sns.despine(ax=axs[i])
        axs[i].set(
            xlim=(xmin, xmax),
            ylim=(ymin, ymax),
            title=' '.join((system, stories, rc, direction)),
        )
        axs[i].plot([0.00, 1.00], [0.00, 1.00], color='black', linewidth=0.30)
    plt.show()
plt.close()

# %% [markdown]
"""
### At what part of the building are PIDs observed?
"""

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
            hue='PIDStoryText',
            ax=axs[i],
            palette={
                'First 2 stories': "#7B9F35",
                'First 3 stories': "#7B9F35",
                'Intermediate stories': "#226666",
                'Last story': "#AA3939",
            },
            s=10,
        )
        sns.despine(ax=axs[i])
        axs[i].set(
            xlim=(xmin, xmax),
            ylim=(ymin, ymax),
            title=' '.join((system, stories, rc, direction)),
        )
    plt.show()
plt.close()


# %% [markdown]
"""
### At what part of the building are RIDs observed?
"""

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
            hue='RIDStoryText',
            ax=axs[i],
            palette={
                'First 2 stories': "#7B9F35",
                'First 3 stories': "#7B9F35",
                'Intermediate stories': "#226666",
                'Last story': "#AA3939",
            },
            s=10,
        )
        sns.despine(ax=axs[i])
        axs[i].set(
            xlim=(xmin, xmax),
            ylim=(ymin, ymax),
            title=' '.join((system, stories, rc, direction)),
        )
    plt.show()
plt.close()


# %% [markdown]
#
