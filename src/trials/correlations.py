"""
Visualize correlations of variables with different model fitting
approaches
"""

# pylint: disable=import-error

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pelicun.assessment import Assessment
from src import models
from src.handle_data import load_dataset
from src.handle_data import remove_collapse
from src.handle_data import only_drifts


def main():
    story_dict = {'i': '2', 'j': '3'}
    model_dict = {}

    num_realizations = 1000

    df_all_edps = remove_collapse(load_dataset('data/edp_extra.parquet')[0])
    df = only_drifts(df_all_edps)

    df_all_edps = (
        df_all_edps.unstack(0)
        .unstack(0)
        .unstack(0)
        .unstack(2)
        .unstack(2)
        .unstack(1)
        .stack(4)
        .dropna(axis=1)
        .drop('PVb', level='edp', axis=1)
        # .drop('RID', level='edp', axis=1)
    )

    for index, story in story_dict.items():
        the_case = ("scbf", "9", "ii", story)  # we combine the two directions
        case_df = df[the_case].dropna().stack(level=0)
        analysis_rid_vals = case_df.dropna()["RID"].to_numpy().reshape(-1)
        analysis_pid_vals = case_df.dropna()["PID"].to_numpy().reshape(-1)

        model_str = models.Model_Bilinear_Weibull()
        model_str.add_data(analysis_pid_vals, analysis_rid_vals)
        model_str.censoring_limit = 0.0025
        model_str.fit(method='mle')

        model_dict[index] = model_str

    # simulate data
    demand_sample_dict = {}
    rid_sample_dfs = []
    for hz in df_all_edps.index.get_level_values(0).unique():
        df_hz = df_all_edps.loc[hz, :]
        # re-index
        df_hz.index = pd.RangeIndex(len(df_hz.index))
        df_hz.columns = pd.MultiIndex.from_tuples(
            [(x[-1], x[-2], '1') for x in df_hz.columns], names=('edp', 'loc', 'dir')
        )
        df_hz.loc['Units', :] = [
            {'PFA': 'inps2', 'PFV': 'inps', 'PID': 'rad', 'RID': 'rad'}[x]
            for x in df_hz.columns.get_level_values('edp')
        ]

        asmt = Assessment({"PrintLog": False, "Seed": 1})
        asmt.stories = 9
        asmt.demand.load_sample(df_hz)
        asmt.demand.calibrate_model(
            {
                "ALL": {
                    "DistributionFamily": "lognormal",
                    "AddUncertainty": 0.00,
                },
                "PID": {
                    "DistributionFamily": "lognormal",
                    "TruncateLower": "",
                    "TruncateUpper": "0.10",
                    "AddUncertainty": 0.00,
                },
            }
        )
        asmt.demand.generate_sample({"SampleSize": num_realizations})
        demand_sample = asmt.demand.save_sample()
        demand_sample_dict[hz] = demand_sample

        # generate RIDs
        num_samples = len(demand_sample['PID', story_dict['i'], '1'].values)
        assert num_samples == len(demand_sample['PID', story_dict['j'], '1'].values)
        uniform_sample = np.random.uniform(0.00, 1.00, num_samples)
        # uniform_sample = None --> No RID-RID correlation
        # pylint: disable=consider-using-dict-items
        for key in story_dict:
            model = model_dict[key]
            model.uniform_sample = uniform_sample
            pids = demand_sample['PID', story_dict[key], '1'].values
            rids = model.generate_rid_samples(pids)
            rid_sample_dfs.append(pd.Series(rids, name=(hz, story_dict[key])))
        # pylint: enable=consider-using-dict-items

    demand_sample_df = pd.concat(
        demand_sample_dict.values(), keys=demand_sample_dict.keys(), names=['hz']
    )

    rid_sample_df = pd.concat(rid_sample_dfs, axis=1)
    rid_sample_df.columns.names = ('hz', 'story')

    # plot PID-PID correlation
    fig, ax = plt.subplots()
    ax.scatter(
        df_all_edps.loc[:, ('scbf', '9', 'ii', story_dict['i'], 'PID')],
        df_all_edps.loc[:, ('scbf', '9', 'ii', story_dict['j'], 'PID')],
        edgecolor='black',
        facecolor='white',
        alpha=0.50,
    )
    ax.scatter(
        demand_sample_df.loc[:, ('PID', story_dict['i'], '1')],
        demand_sample_df.loc[:, ('PID', story_dict['j'], '1')],
        edgecolor='blue',
        facecolor='white',
        alpha=0.05,
        s=1,
    )
    ax.plot([0.00, 1.00], [0.00, 1.00], linestyle='dashed', color='black')
    ax.set(xlim=(0.00, 0.1), ylim=(0.00, 0.1))
    ax.grid(which='both', linewidth=0.30)
    ax.set(xlabel=f'PID, story {story_dict["i"]}')
    ax.set(ylabel=f'PID, story {story_dict["j"]}')
    plt.show()

    # plot RID-RID correlation
    plt.close()
    fig, ax = plt.subplots()
    ax.scatter(
        demand_sample_df.loc[:, ('RID', story_dict['i'], '1')].values[::10],
        demand_sample_df.loc[:, ('RID', story_dict['j'], '1')].values[::10],
        edgecolor='blue',
        facecolor='white',
        alpha=0.50,
        label='direct fit',
    )
    ax.scatter(
        df.stack(level=4).loc[:, ('scbf', '9', 'ii', story_dict['i'], 'RID')],
        df.stack(level=4).loc[:, ('scbf', '9', 'ii', story_dict['j'], 'RID')],
        edgecolor='black',
        facecolor='white',
        label='empirical',
    )
    ax.scatter(
        rid_sample_df.xs(story_dict['i'], level='story', axis=1)
        .stack()
        .values[::10],
        rid_sample_df.xs(story_dict['j'], level='story', axis=1)
        .stack()
        .values[::10],
        edgecolor='red',
        facecolor='white',
        alpha=0.50,
        label='conditional Weibull',
    )
    ax.plot([0.00, 1.00], [0.00, 1.00], linestyle='dashed', color='black')
    ax.set(xlim=(0.00, 0.08), ylim=(0.00, 0.08))
    ax.set(xlabel=f'RID, story {story_dict["i"]}')
    ax.set(ylabel=f'RID, story {story_dict["j"]}')
    ax.grid(which='both', linewidth=0.30)
    plt.legend()
    plt.show()

    # plot RID-PID correlation
    plt.close()
    fig, ax = plt.subplots()
    ax.scatter(
        demand_sample_df.loc[:, ('RID', story_dict['i'], '1')].values[::10],
        demand_sample_df.loc[:, ('PID', story_dict['i'], '1')].values[::10],
        edgecolor='blue',
        facecolor='white',
        alpha=0.50,
        label='direct fit',
    )
    ax.scatter(
        rid_sample_df.loc[:, story_dict['i']]
        .reorder_levels([1, 0])
        .sort_index()
        .values[::10],
        demand_sample_df.loc[:, ('PID', story_dict['i'], '1')].values[::10],
        edgecolor='red',
        facecolor='white',
        alpha=0.50,
        label='conditional Weibull',
    )
    ax.scatter(
        df.stack(level=4).loc[:, ('scbf', '9', 'ii', story_dict['i'], 'RID')],
        df_all_edps.loc[:, ('scbf', '9', 'ii', story_dict['i'], 'PID')],
        edgecolor='black',
        facecolor='white',
        label='empirical',
    )
    ax.plot([0.00, 1.00], [0.00, 1.00], linestyle='dashed', color='black')
    ax.set(xlim=(0.00, 0.08), ylim=(0.00, 0.08))
    ax.set(xlabel=f'PID, story {story_dict["i"]}')
    ax.set(ylabel=f'RID, story {story_dict["i"]}')
    ax.grid(which='both', linewidth=0.30)
    plt.legend()
    plt.show()

    hz = '15'
    fig, ax = plt.subplots()
    sns.ecdfplot(
        df.stack(level=4).loc[hz, ('scbf', '9', 'ii', story_dict['i'], 'RID')],
        label='empirical',
    )
    sns.ecdfplot(
        rid_sample_df.loc[hz, story_dict['i']],
        label='conditional Weibull',
    )
    sns.ecdfplot(
        demand_sample_df.loc[hz, ('RID', story_dict['i'], '1')], label='direct fit'
    )
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
