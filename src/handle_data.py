"""
Functions to process the input data
"""

import pandas as pd


def load_dataset(path='data/edp_cs.parquet') -> tuple[pd.Series, dict[str, str]]:
    """
    Load the analysis results and their units.
    """

    df = pd.read_parquet(path)

    # turn back into a pd.Series
    # (was stored as DataFrame to use df.to_parquet())
    series = df['value']

    units = {'PFA': 'g', 'PID': 'rad', 'PFV': 'in/s', 'PVb': 'lb'}

    return series, units


def remove_collapse(data: pd.Series, drift_threshold: float = 0.10) -> pd.Series:
    """
    Remove collapse instances
    """

    # drift_threshold = 0.10  # that's 10%

    initial_level_order = data.index.names

    # shuffle columns/rows to get a different view
    # (this doesn't create a copy)
    df_unstack = data.unstack(4).unstack(4).unstack(4)

    # globally remove cmp-loc-dir entries that correspond to a
    # collapse
    pid_df = df_unstack['PID'] > drift_threshold
    not_collapse_bool_index = ~pid_df.apply(any, axis=1)

    df_unstack_no_collapse = df_unstack.loc[not_collapse_bool_index, :]

    df_no_collapse = (
        df_unstack_no_collapse.stack()
        .stack()
        .stack()
        .reorder_levels(initial_level_order)
    )

    assert isinstance(df_no_collapse, pd.Series)
    return df_no_collapse


def only_drifts(df_no_collapse: pd.Series) -> pd.DataFrame:
    filtered_df = (
        df_no_collapse[
            df_no_collapse.index.get_level_values('edp').isin(("PID", "RID"))
        ]
        .unstack(0)
        .unstack(0)
        .unstack(0)
        .unstack(2)
        .unstack(2)
        .unstack(1)
    )
    assert isinstance(filtered_df, pd.DataFrame)
    return filtered_df
