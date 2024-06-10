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
# Fitting collapse fragilities to the structural analysis results

`Wed May  8 05:29:27 AM PDT 2024`
"""

# %% [markdown]
# """
# ## Setup
# """
# Change directory to project's root, if needed
# (not needed when re-evaluating)
# from pathlib import Path
# import os, sys
# if Path(os.getcwd()).name != '2023-11_RID_Realizations':
#     os.chdir('../../../../')
# sys.path.append('src/notebooks/exploratory_data_analysis/2024-05-08_collapse_fragilities')


# %%
# Imports
import os
import sys
from pathlib import Path

# Change directory to project's root, if needed
# (not needed when re-evaluating)
if Path(os.getcwd()).name != '2023-11_RID_Realizations':
    os.chdir('../../../../')

sys.path.append(
    'src/notebooks/exploratory_data_analysis/2024-05-08_collapse_fragilities'
)

from itertools import product
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm
from imports import get_collapse_data
from imports import get_collapse_probabilities
from imports import get_sa
from imports import neg_log_likelihood


# %%
df = pd.read_parquet('data/edp_cs.parquet')
df.index = df.index.reorder_levels(['archetype', 'dir', 'edp', 'hz', 'gm', 'loc'])
df = df.sort_index()

# %%
# Get the collapse probabilities for each archetype
index = []
probs = []
for system, stories, rc in product(
    ('smrf', 'scbf', 'brbf'), ('3', '6', '9'), ('ii', 'iv')
):
    index.append((system, stories, rc))
    probs.append(
        get_collapse_probabilities(*get_collapse_data(df, system, stories, rc, 0.08))
    )
prob_df = pd.concat(probs, axis=1, keys=index)
print((prob_df * 100.00).T.to_string(float_format='{:.1f}'.format))


# %%
index = []
parameters = []
for system, stories, rc in product(
    ('smrf', 'scbf', 'brbf'), ('3', '6', '9'), ('ii', 'iv')
):

    index.append((system, stories, rc))

    count, num_collapses = get_collapse_data(df, system, stories, rc, 0.05)
    for val in count.index:
        if val not in num_collapses.index:
            num_collapses.loc[val] = 0
    num_collapses = num_collapses.sort_index()

    xjs = np.array([get_sa(system, stories, rc, hz) for hz in count.index])
    njs = count.values
    zjs = num_collapses.values

    x0 = np.array((3.00, 0.40))
    res = minimize(
        neg_log_likelihood,
        x0,
        method="nelder-mead",
        args=(njs, zjs, xjs),
        # bounds=((0.0, 100.00), (0.40, 0.40)),
    )

    parameters.append((res.x[0], res.x[1]))

frag_df = pd.DataFrame(
    parameters,
    index=pd.MultiIndex.from_tuples(index),
    columns=['Theta_0', 'Theta_1'],
).rename_axis(index=['system', 'stories', 'rc'])
frag_df

# %%

frag_df.to_parquet(
    'src/notebooks/exploratory_data_analysis/'
    '2024-05-08_collapse_fragilities/collapse_fragilities.parquet'
)

# %% [markdown]
"""
We also need a dataframe containing Sa(T1) for each hazard level.

"""

# %%
sa_values = {}
for system, stories, rc, hz in product(
    ('smrf', 'scbf', 'brbf'),
    ('3', '6', '9'),
    ('ii', 'iv'),
    [f'{i + 1}' for i in range(29)],
):
    sa = get_sa(system, stories, rc, hz)
    sa_values[(system, stories, rc, int(hz))] = sa
sa_df = (
    pd.Series(sa_values.values(), index=sa_values.keys(), name='Sa(T1)')
    .rename_axis(index=['system', 'stories', 'rc', 'hz'])
    .unstack('hz')
)
sa_df.to_parquet(
    'src/notebooks/exploratory_data_analysis/'
    '2024-05-08_collapse_fragilities/sa_t1.parquet'
)

# %% [markdown]
"""
We calculate the probability of collapse by evaluating the collapse
fragilities.

"""


def calculate_probability(system: str, stories: str, rc: str, hz: int) -> float:
    sa_value = sa_values[(system, stories, rc, hz)]
    delta, beta = frag_df.loc[system, stories, rc]
    z = (np.log(sa_value) - np.log(delta)) / beta
    probability = norm.cdf(z)
    return probability


probabilities = {}
for system, stories, rc, hz in product(
    ('smrf', 'scbf', 'brbf'),
    ('3', '6', '9'),
    ('ii', 'iv'),
    [f'{i + 1}' for i in range(29)],
):
    probabilities[system, stories, rc, int(hz)] = calculate_probability(
        system, stories, rc, int(hz)
    )
probability_df = (
    pd.Series(probabilities.values(), index=probabilities.keys(), name='P(C)')
    .rename_axis(index=['system', 'stories', 'rc', 'hz'])
    .unstack('hz')
)
probability_df.to_parquet(
    'src/notebooks/exploratory_data_analysis/'
    '2024-05-08_collapse_fragilities/pr_collapse.parquet'
)


# %%
