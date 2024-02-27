"""
Produce plots to visually assess the quality of the fit to PID-RID
data for all considered archetypes.
"""

import concurrent.futures
from itertools import product
import pickle
import tqdm
import pandas as pd
import matplotlib.pyplot as plt
from src.util import store_info


def generate_figure(system, stories, rc, model_df, method, data_gathering_approach):
    """
    Generate a figure showing the PID-RID pairs, their rolling
    quantiles and the fitted model's quantiles.  The figure is laid
    out in groups of two columns (x-dir, y-dir) and three rows
    (three stories). Taller archetypes (6-story, 9-story) contain
    additional groups on the right of the previous one.

    """

    df = model_df[system][stories][rc]
    num_stories = int(stories)

    num_column_pairs = int(num_stories / 3)

    if data_gathering_approach == 'separate_directions':
        num_columns = 2 * num_column_pairs
    else:
        num_columns = num_column_pairs
    num_rows = 3  # always the case
    box_w, box_h = 1.5, 1.5  # inch

    if data_gathering_approach == 'separate_directions':
        drc = {'1': 'x', '2': 'y'}

    plt.close()
    fig, axs = plt.subplots(
        num_rows,
        num_columns,
        figsize=(box_h * num_columns, box_w * num_rows),
        sharex=True,
        sharey=True,
    )
    axs = axs.reshape(3, -1)
    for i_col in range(num_columns):
        for i_row in range(num_rows):
            if data_gathering_approach == 'separate_directions':
                group = int(i_col / 2)
                story = str(i_row + 1 + 3 * group)
                direction = str(i_col % 2 + 1)
                model = df[(story, direction)]
            else:
                group = i_col
                story = str(i_row + 1 + 3 * group)
                model = df[story]
            model.plot_model(axs[i_row, i_col])
            text = '\n'.join(
                [
                    f'{y}={x:.5f}'
                    for x, y in zip(model.parameters, model.parameter_names)
                ]
            )
            text += '\n' + f'LL={-model.fit_meta.fun:.2e}'
            axs[i_row, i_col].text(
                0.98,
                0.02,
                text,
                horizontalalignment='right',
                verticalalignment='bottom',
                transform=axs[i_row, i_col].transAxes,
                fontsize=4,
            )
            if data_gathering_approach == 'separate_directions':
                text = f'{story}{drc[direction]}'
            else:
                text = f'{story}'
            axs[i_row, i_col].text(
                0.05,
                0.95,
                text,
                horizontalalignment='left',
                verticalalignment='top',
                transform=axs[i_row, i_col].transAxes,
                bbox={'facecolor': 'white', 'alpha': 0.5},
                fontsize=6,
            )
            axs[i_row, i_col].set(xlim=(-0.005, 0.025), ylim=(-0.005, 0.05))
    fig.suptitle(f'{stories.upper()}-STORY RC {rc.upper()} {system.upper()}')
    fig.text(0.5, 0.01, 'Residual Inter-story Drift (RID)', ha='center')
    fig.text(
        0.01, 0.50, 'Peak Inter-story Drift (PID)', va='center', rotation='vertical'
    )
    if data_gathering_approach == 'separate_directions':
        left_adj = {'3': 0.17, '6': 0.10, '9': 0.07}
    else:
        left_adj = {'3': 0.17 * 2.0, '6': 0.10 * 2.0, '9': 0.07 * 2.0}
    fig.subplots_adjust(
        wspace=0.00,
        hspace=0.00,
        top=0.94,
        left=left_adj[stories],
        right=0.98,
        bottom=0.08,
    )
    # plt.show()
    models_path = (
        f'results/parameters/{data_gathering_approach}/{method}/models.pickle'
    )
    plt.savefig(
        store_info(
            f'results/figures/{data_gathering_approach}'
            f'/{method}/fit_{system}_{stories}_{rc}.pdf',
            [models_path],
        )
    )
    plt.savefig(
        store_info(
            f'results/figures/{data_gathering_approach}'
            f'/{method}/fit_{system}_{stories}_{rc}.svg',
            [models_path],
        )
    )
    plt.close()


def main():
    methods = ('weibull_bilinear', 'gamma_bilinear',)
    data_gathering_approaches = ('separate_directions', 'bundled_directions')

    model_dfs = {}
    for method, data_gathering_approach in product(
        methods, data_gathering_approaches
    ):
        models_path = (
            f'results/parameters/{data_gathering_approach}/'
            f'{method}/models.pickle'
        )
        with open(models_path, 'rb') as f:
            models = pickle.load(f)

        # turn the dict into a dataframe for more convenient indexing
        # operations
        model_df = pd.Series(models.values(), index=models.keys())
        if data_gathering_approach == 'separate_directions':
            model_df.index.names = ('system', 'stories', 'rc', 'loc', 'dir')
        else:
            model_df.index.names = ('system', 'stories', 'rc', 'loc')
        model_dfs[(method, data_gathering_approach)] = model_df

    args_list = []
    for system, stories, rc, method, data_gathering_approach in list(
        product(
            ('smrf', 'scbf', 'brbf'),
            ('3', '6', '9'),
            ('ii', 'iv'),
            methods,
            data_gathering_approaches,
        )
    ):
        args_list.append(
            (
                system,
                stories,
                rc,
                model_dfs[method, data_gathering_approach],
                method,
                data_gathering_approach,
            )
        )
    for args in tqdm.tqdm(args_list):
        generate_figure(*args)


if __name__ == '__main__':
    main(parallel=False)
