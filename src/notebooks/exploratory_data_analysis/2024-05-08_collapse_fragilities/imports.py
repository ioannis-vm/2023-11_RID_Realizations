"""
Improtable objects for collapse_fragilities.py

"""

import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.special import binom
from scipy.interpolate import interp1d
from extra.structural_analysis.src.util import read_study_param


def get_collapse_data(df, system, stories, rc, threshold):
    """
    Obtains the number of available records for each hazard level and
    how many of those correspond to collapse.

    Collapse is identified as the maximum residual drift of all
    stories exceeding some threshold value.

    Parameters
    ----------
    df: pd.DataFrame
        Dataframe with all the results.
    system: str
        Any of {smrf, scbf, brbf}
    stories: str
        Any of {3, 6, 9}
    rc: str
        Any of {ii, iv}
    threshold: float
        Residual drift above which we consider the case to ba a
        collapse, in [rad] units.

    Returns
    -------
    tuple
        Contains two pandas Series, one with the number of records and
        another with the number of collapses.

    """
    archetype = f'{system}_{stories}_{rc}'
    # subset based on above parameters
    df_archetype = df.loc[(archetype, slice(None), 'PID'), 'value']
    # drop levels where the value is always the same
    df_archetype.index = df_archetype.index.droplevel(
        ['archetype', 'edp']
    )
    # unstack directions
    df_archetype = df_archetype.unstack('dir')
    # reorder levels
    lvls = ['hz', 'gm', 'loc']
    df_archetype.index = df_archetype.index.reorder_levels(lvls)

    # if a record is missing in one direction but exists in the other,
    # we use the existing for both. If both are missing, we drop the
    # row.
    df_archetype['x'] = df_archetype['x'].fillna(df_archetype['y'])
    df_archetype['y'] = df_archetype['y'].fillna(df_archetype['x'])
    df_archetype.dropna(subset=['x', 'y'], how='all', inplace=True)

    # take the maximum drift observed in the two directions and then
    # at each story
    df_max_drift = df_archetype.max(axis=1).unstack('loc').max(axis=1)

    # count number of available ground motions for each hz
    count = df_max_drift.groupby(by='hz').count()
    count.index = count.index.astype('int')
    count = count.sort_index()

    # count collapses
    collapse_cases = df_max_drift[df_max_drift > threshold]
    num_collapses = collapse_cases.groupby(by='hz').count()
    num_collapses.index = num_collapses.index.astype('int')
    num_collapses = num_collapses.sort_index()

    return count, num_collapses


def get_collapse_probabilities(count, num_collapses):
    """
    Obtains the collapse probability for each hazard level for a given
    archetype using the available data.

    Collapse is identified as the maximum residual drift of all
    stories exceeding some threshold value.

    Parameters
    ----------
    count: pd.Series
        Number of available records for each hazard level.
    num_collapses: pd.Series
        Number of collapse cases among those records.

    Returns
    -------
    pd.Series
        Pandas series containing the collapse probability for each
        hazard level.

    """

    collapse_probability = (num_collapses / count).fillna(0.00)
    collapse_probability.index = collapse_probability.index.astype(int)
    collapse_probability = collapse_probability.sort_index()

    return collapse_probability


def neg_log_likelihood(x, njs, zjs, xjs):
    """
    Calculates the negative log likelihood of observing the given data
    under the specified distribution parameters.
    """
    theta, beta = x
    phi = norm.cdf(np.log(xjs / theta) / beta)
    logl = np.sum(
        np.log(binom(njs, zjs))
        + zjs * np.log(phi)
        + (njs - zjs) * np.log(1.00 - phi)
    )
    return -logl


def get_sa(system, stories, rc, hz):
    """
    Get the Sa(T1) that corresponds to a hazard level for a given
    archetype.
    system: str
        Any of {smrf, scbf, brbf}
    stories: str
        Any of {3, 6, 9}
    rc: str
        Any of {ii, iv}
    hz: str
        Any of {'1', ..., '25'}

    """
    spectrum = pd.read_csv(
        f'extra/structural_analysis/results/site_hazard/UHS_{hz}.csv',
        index_col=0,
        header=0,
    )
    base_period = float(
        read_study_param(
            f'extra/structural_analysis/data/{system}_{stories}_{rc}/period_closest'
        )
    )
    ifun = interp1d(spectrum.index.to_numpy(), spectrum.to_numpy().reshape(-1))
    sa = float(ifun(base_period))
    return sa
