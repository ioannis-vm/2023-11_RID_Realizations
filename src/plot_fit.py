"""
Produce plots to visually assess the quality of the fit to PID-RID
data for all considered archetypes.
"""

from itertools import product
import pickle
import tqdm
import pandas as pd
import matplotlib.pyplot as plt
from src.util import store_info


def generate_figure(system, stories, rc, model_df, method, models_path):
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

    num_columns = 2 * num_column_pairs
    num_rows = 3  # always the case
    box_w, box_h = 1.5, 1.5  # inch

    drc = {'1': 'x', '2': 'y'}

    plt.close()
    fig, axs = plt.subplots(
        num_rows,
        num_columns,
        figsize=(box_h * num_columns, box_w * num_rows),
        sharex=True,
        sharey=True,
    )
    for i_col in range(num_columns):
        for i_row in range(num_rows):
            group = int(i_col / 2)
            story = str(i_row + 1 + 3 * group)
            direction = str(i_col % 2 + 1)
            model = df[(story, direction)]
            model.plot_model(axs[i_row, i_col])
            text = '\n'.join([f'{x:.5f}' for x in model.parameters])
            axs[i_row, i_col].text(
                0.98,
                0.02,
                text,
                horizontalalignment='right',
                verticalalignment='bottom',
                transform=axs[i_row, i_col].transAxes,
                fontsize=6,
            )
            text = f'{story}{drc[direction]}'
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
            # axs[i_row, i_col].fill_between(
            #     (0.002, 0.015), -1, 1, alpha=0.10, color='black'
            # )
            axs[i_row, i_col].set(xlim=(-0.005, 0.025), ylim=(-0.005, 0.05))
    fig.suptitle(f'{stories.upper()}-STORY RC {rc.upper()} {system.upper()}')
    fig.text(0.5, 0.01, 'Residual Inter-story Drift (RID)', ha='center')
    fig.text(
        0.01, 0.50, 'Peak Inter-story Drift (PID)', va='center', rotation='vertical'
    )
    left_adj = {'3': 0.17, '6': 0.10, '9': 0.07}
    fig.subplots_adjust(
        wspace=0.00,
        hspace=0.00,
        top=0.94,
        left=left_adj[stories],
        right=0.98,
        bottom=0.08,
    )
    # plt.show()
    plt.savefig(
        store_info(
            f'results/figures/{method}/fit_{system}_{stories}_{rc}.pdf',
            [models_path],
        )
    )
    plt.close()


def main():
    # load models
    method = 'weibull_bilinear'
    # method = 'gamma_bilinear'
    # method = 'beta_bilinear'
    models_path = f'results/parameters/{method}/models.pickle'
    with open(models_path, 'rb') as f:
        models = pickle.load(f)

    # turn the dict into a dataframe for more convenient indexing
    # operations
    model_df = pd.Series(models.values(), index=models.keys())
    model_df.index.names = ('system', 'stories', 'rc', 'loc', 'dir')

    for system, stories, rc in tqdm.tqdm(
        list(product(('smrf', 'scbf', 'brbf'), ('3', '6', '9'), ('ii', 'iv')))
    ):
        generate_figure(system, stories, rc, model_df, method, models_path)


if __name__ == '__main__':
    main()
