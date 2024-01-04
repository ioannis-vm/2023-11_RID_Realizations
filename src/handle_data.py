"""
Functions to process the input data
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def load_dataset():
    """
      Load the analysis results and their units.
    """
    df = pd.read_parquet('data/edp.parquet')

    # turn back into a pd.Series
    # (was stored as DataFrame to use df.to_parquet())
    df = df['value']

    units = {
        'PFA': 'g',
        'PID': 'rad',
        'PFV': 'in/s',
        'PVb': 'lb'
    }

    return df, units


def remove_collapse(df):
    """
      Remove collapse instances
    """

    drift_threshold = 0.08      # that's 8%

    initial_level_order = df.index.names

    # shuffle columns/rows to get a different view
    # (this doesn't create a copy)
    df_unstack = df.unstack(4).unstack(4).unstack(4)

    # globally remove cmp-loc-dir entries that correspond to a
    # collapse
    pid_df = df_unstack['PID'] > drift_threshold
    not_collapse_bool_index = ~pid_df.apply(any, axis=1)

    df_unstack_no_collapse = (
        df_unstack.loc[not_collapse_bool_index, :])
    df_unstack = (df.unstack()
        .unstack(0).unstack(0)
        .unstack(0).unstack(0)
        .reorder_levels(
            ('system', 'stories', 'rc', 'hz', 'gm'), axis=1)
        )

    df_no_collapse = (
        df_unstack_no_collapse
        .stack().stack().stack()
        .reorder_levels(initial_level_order)
    )

    return df_no_collapse


def only_drifts(df_no_collapse):

    filtered_df = (
        df_no_collapse[
            df_no_collapse.index
            .get_level_values('edp')
            .isin(("PID", "RID"))]
        .unstack(0)
        .unstack(0)
        .unstack(0)
        .unstack(2)
        .unstack(2)
        .unstack(1)
    )
    return filtered_df


def scatter_pid_rid(filtered_df):

    # we fit each archetype-story-direction individually
    archetype = ("smrf", "9", "ii", "2", "2")
    # archetype = ("smrf", "3", "ii")

    archetype_df = filtered_df[archetype]

    # archetype_df = filtered_df

    # archetype_df = archetype_df.stack(0).stack(0).stack(0).stack(0).stack(0)

    _, ax = plt.subplots()
    sns.scatterplot(
        data=archetype_df,
        x="RID",
        y="PID",
        hue=archetype_df.index.get_level_values("hz"),
        ax=ax,
    )
    ax.set(xlim=(-0.005, 0.04), ylim=(-0.005, 0.08))
    ax.grid(which="both", linewidth=0.30)
    plt.show()
