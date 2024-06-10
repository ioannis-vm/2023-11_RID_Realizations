"""
Explore fitting various models to the data.

"""

from itertools import product
from collections import defaultdict
from scipy.stats import norm
from tqdm import tqdm
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots
import plotly.subplots as sp
from src import models


def get_pid_rid_pairs(df, system, stories, rc, story):
    """
    Retrieves PID-RID pairs for the given case.

    Parameters
    ----------
    df: pd.DataFrame
        Dataframe with EDP results.
    system: str
        Any of {smrf, scbf, brbf}
    stories: str
        Any of {3, 6, 9}
    rc: str
        Any of {ii, iv}
    direction: str
        Any of {x, y}
    story: str
        Any of {1, 2, ..., stories} or 'max-max'

    Returns
    -------
    tuple or pd.DataFrame
        If a location is given, returns:
        - Dataframe with PID-RID pairs, with unstable collapse cases
        removed (transient drift exceeding 10%).
        - RSN of each pair.
        - Applied scaling factor.
        If the location is max-max:
        - Dataframe with maxPID-maxRID pairs, with unstable collapse
        cases removed (transient drift exceeding 10%), and
        descriptions of the story where the max values occurred.

    """

    df_archetype = df.loc[(system, stories, rc), :]

    # remove cases where PID > 0.10 (unstable collapse)
    pairs_all_stories = (
        df_archetype.loc[
            slice(None),
            ['PID', 'RID'],
            slice(None),
            slice(None),
            slice(None),
        ]
        .unstack('edp')
        .unstack('loc')
    )
    collapse_bool = (pairs_all_stories[('value', 'PID')] > 0.10).any(axis=1).values
    pairs_all_stories = pairs_all_stories.iloc[~collapse_bool, :]

    if story == 'max-max':
        pid = {}
        pid_story = {}
        rid = {}
        rid_story = {}
        for direction in ('x', 'y'):
            pid[direction] = pairs_all_stories.loc[direction, ('value', 'PID')]
            pid[direction].columns = pid[direction].columns.astype(int)
            pid_story[direction] = pid[direction].idxmax(axis=1)
            pid[direction] = pid[direction].max(axis=1)
            rid[direction] = pairs_all_stories.loc[direction, ('value', 'RID')]
            rid[direction].columns = rid[direction].columns.astype(int)
            rid_story[direction] = rid[direction].idxmax(axis=1)
            rid[direction] = rid[direction].max(axis=1)
        pid = pd.concat((pid['x'], pid['y']), keys=('x', 'y'))
        rid = pd.concat((rid['x'], rid['y']), keys=('x', 'y'))
        pid_story = pd.concat((pid_story['x'], pid_story['y']), keys=('x', 'y'))
        rid_story = pd.concat((rid_story['x'], rid_story['y']), keys=('x', 'y'))

        pairs = pd.concat(
            (pid, rid, pid_story, rid_story),
            keys=['PID', 'RID', 'PIDStory', 'RIDStory'],
            axis=1,
        )
        pairs.index.names = ['dir', *pairs.index.names[1:]]
        pairs = pairs.reset_index()

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

    # filter story
    pairs = pairs_all_stories.iloc[
        :, pairs_all_stories.columns.get_level_values('loc') == story
    ]
    pairs.columns = pairs.columns.droplevel('loc')

    pairs = pairs.dropna(how='any')
    rsns = pairs['rsn']['PID']
    scaling = pairs['scaling_factor']['PID']
    pairs = pairs.loc[:, 'value']
    pairs = pairs.reset_index()

    return pairs, rsns, scaling


def generate_plot(
    df,
    df_collapse,
    selected_model,
    system,
    stories,
    rc,
    story,
    criterion,
    rid_capacity_delta,
    rid_capacity_beta,
    xmin,
    xmax,
    ymin,
    ymax,
    show_collapse_probability,
):

    if story == 'max-max':
        pairs = get_pid_rid_pairs(df, system, stories, rc, story)
    else:
        pairs, rsn_data, scale_data = get_pid_rid_pairs(
            df, system, stories, rc, story
        )

    roll = models.Model()
    roll.add_data(pairs['PID'].values, pairs['RID'].values)
    roll.calculate_rolling_quantiles()

    pelicun_fitted_model = selected_model
    weibull_subset = pairs
    # weibull_subset = weibull_subset[weibull_subset['PID'] > 0.02]
    pids = weibull_subset['PID'].values
    rids = weibull_subset['RID'].values
    pelicun_fitted_model.add_data(pids, rids)
    pelicun_fitted_model.censoring_limit = 0.00025
    pelicun_fitted_model.fit(method='mle')

    model_pid = np.linspace(0.00, 0.06, 1000)
    model_rid_50 = pelicun_fitted_model.evaluate_inverse_cdf(0.50, model_pid)
    model_rid_20 = pelicun_fitted_model.evaluate_inverse_cdf(0.20, model_pid)
    model_rid_80 = pelicun_fitted_model.evaluate_inverse_cdf(0.80, model_pid)

    c0 = '#7C296A'
    # c1 = '#256F5B'
    # c2 = '#AA9139'

    # Probability of excessive drift
    pid_conditioning_values = np.arange(0.0025, 0.08, 0.0025)
    halfwidth = 0.002
    prob_empirical = []
    prob_weibull = []

    num_realizations = 10000
    for i, val in enumerate(pid_conditioning_values):
        subset_pairs = pairs
        subset_pairs = subset_pairs[subset_pairs['PID'] > val - halfwidth]
        subset_pairs = subset_pairs[subset_pairs['PID'] < val + halfwidth]
        if len(subset_pairs) < 20:
            prob_empirical.append(np.nan)
            prob_weibull.append(np.nan)
        else:
            pid_array = np.random.choice(
                subset_pairs['PID'].values, size=num_realizations, replace=True
            )
            capacities = rid_capacity_delta * np.exp(
                rid_capacity_beta * norm.rvs(size=num_realizations)
            )
            rids_empirical = np.random.choice(
                subset_pairs['RID'].values, size=num_realizations, replace=True
            )
            rids_weibull = pelicun_fitted_model.generate_rid_samples(pid_array)
            prob_empirical.append(
                sum(rids_empirical > capacities) / float(num_realizations)
            )
            prob_weibull.append(
                sum(rids_weibull > capacities) / float(num_realizations)
            )

    if show_collapse_probability:
        ncols = 3
        column_widths = [0.5, 0.25, 0.25]
        subplot_titles = (
            f'{system}_{stories}_{rc} story {story} RID|PID distribution',
            'Probability of DS1',
            'Probability of collapse',
        )
    else:
        ncols = 2
        column_widths = [0.7, 0.3]
        subplot_titles = (
            f'{system}_{stories}_{rc} story {story} RID|PID distribution',
            'Probability of DS1',
        )

    fig = make_subplots(
        rows=1,
        cols=ncols,
        shared_yaxes=True,
        column_widths=column_widths,
        subplot_titles=subplot_titles,
    )

    if story == 'max-max':
        categories = pairs[criterion].unique()

        palette = {
            'Same story': "#7B9F35",
            '1--2 stories': "#226666",
            '>2 stories': "#AA3939",
            'First 2 stories': "#7B9F35",
            'First 3 stories': "#7B9F35",
            'Intermediate stories': "#226666",
            'Last story': "#226666",
        }

        for category in categories:

            custom_data = list(
                zip(
                    pairs[pairs[criterion] == category]['dir'],
                    pairs[pairs[criterion] == category]['hz'],
                    pairs[pairs[criterion] == category]['PIDStory'],
                    pairs[pairs[criterion] == category]['RIDStory'],
                )
            )
            hovertemplate = (
                'RID: %{x}<br>'
                'PID: %{y}<br>'
                'DIR: %{customdata[0]}<br>'
                'HZLVL: %{customdata[1]}<br>'
                'PID Story: %{customdata[2]}<br>'
                'RID Story: %{customdata[3]}<extra></extra>'
            )

            fig.add_trace(
                go.Scatter(
                    x=pairs[pairs[criterion] == category]['RID'],
                    y=pairs[pairs[criterion] == category]['PID'],
                    mode='markers',
                    marker=dict(
                        color=palette[category],
                    ),
                    name=category,
                    customdata=custom_data,
                    hovertemplate=hovertemplate,
                ),
                row=1,
                col=1,
            )

    else:
        custom_data = list(zip(pairs['dir'], pairs['hz'], rsn_data, scale_data))
        hovertemplate = (
            'RID: %{x}<br>'
            'PID: %{y}<br>'
            'DIR: %{customdata[0]}<br>'
            'HZLVL: %{customdata[1]}<br>'
            'RSN: %{customdata[2]}<br>'
            'SF: %{customdata[3]}<extra></extra>'
        )

        fig.add_trace(
            go.Scatter(
                x=pairs['RID'],
                y=pairs['PID'],
                mode='markers',
                marker=dict(
                    symbol='cross-thin-open',
                    color='gray',
                    size=6,
                    opacity=0.90,
                ),
                name='Scatter Data',
                customdata=custom_data,
                hovertemplate=hovertemplate,
            ),
            row=1,
            col=1,
        )

    empirical_data = [
        (roll.rolling_rid_50, roll.rolling_pid, 3.5, True),
        (roll.rolling_rid_20, roll.rolling_pid, 1.5, False),
        (roll.rolling_rid_80, roll.rolling_pid, 1.5, False),
    ]

    cw_data = [
        (model_rid_50, model_pid, 3.5, True),
        (model_rid_20, model_pid, 1.5, False),
        (model_rid_80, model_pid, 1.5, False),
    ]

    def add_traces(data, color, name, legendgroup):
        for x, y, width, showlegend in data:
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=y,
                    mode='lines',
                    line={'color': color, 'width': width},
                    name=name if showlegend else None,
                    showlegend=showlegend,
                    legendgroup=legendgroup,
                ),
                row=1,
                col=1,
            )

    add_traces(empirical_data, 'black', 'Empirical rolling quantiles', 'erq')
    add_traces(cw_data, c0, 'Conditional Weibull', 'cw')

    # Shade censoring limit region
    fig.add_shape(
        type="rect",
        x0=0.00,
        y0=ymin,
        x1=pelicun_fitted_model.censoring_limit,
        y1=ymax,
        fillcolor=c0,
        opacity=0.1,
        layer="below",
        line_width=0,
        row=1,
        col=1,
    )

    fig.update_xaxes(range=[xmin, xmax], row=1, col=1)
    fig.update_yaxes(range=[ymin, ymax], row=1, col=1)
    fig.update_xaxes(range=[-0.02, 1.02], row=1, col=2)
    fig.update_yaxes(range=[ymin, ymax], row=1, col=2)

    for xvals, name, color, legendgroup in zip(
        (prob_empirical, prob_weibull),
        ('Empirical', 'Conditional Weibull'),
        ('black', c0),
        ('erq', 'cw'),
    ):

        fig.add_trace(
            go.Scatter(
                x=xvals,
                y=pid_conditioning_values,
                mode='lines+markers',
                name=name,
                marker=dict(color=color),
                line=dict(color=color),
                legendgroup=legendgroup,
            ),
            row=1,
            col=2,
        )

    if show_collapse_probability:
        # retrieve collapse probability
        collapse_probability = df_collapse.loc[system, stories, rc]
        collapse_probability.name = 'P(C)'
        # get the mean PID for each hazard level
        median_pid = pairs[['hz', 'PID']].astype({'hz': int}).groupby('hz').median()
        # combine to form a single dataframe and remove NaNs
        pid_and_collapse = pd.concat(
            (median_pid, collapse_probability), axis=1
        ).dropna(how='any')
        fig.add_trace(
            go.Scatter(
                x=pid_and_collapse['P(C)'],
                y=pid_and_collapse['PID'],
                mode='lines+markers',
                name='P(Collapse)',
                marker=dict(color='black'),
                line=dict(color='black'),
                # legendgroup=legendgroup,
            ),
            row=1,
            col=3,
        )

    # Update layout to position the legend at the top
    fig.update_layout(
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=900,
    )

    fig.update_xaxes(title_text="RID", row=1, col=1)
    fig.update_yaxes(title_text="PID", row=1, col=1)
    fig.update_xaxes(title_text="P(excessive drift)", row=1, col=2)
    if show_collapse_probability:
        fig.update_xaxes(title_text="P(collapse)", row=1, col=3)
        fig.update_xaxes(range=[-0.02, 1.02], row=1, col=3)

    fig.show()


def main():

    selected_model = models.Model_Weibull_Trilinear()
    # selected_model = models.Model_1_Weibull()

    # Structural system:
    # any of {'smrf', 'scbf', 'brbf'}

    # Total number of stories:
    # any of {'3', '6', '9'}

    # Design risk category:
    # any of {'ii', 'iv'}

    # Story:
    # any of {'1', '2', ...} or 'max-max'

    # system, stories, rc, story = 'smrf', '3', 'ii', '1'
    # system, stories, rc, story = 'smrf', '3', 'ii', '2'
    # system, stories, rc, story = 'smrf', '3', 'ii', '3'

    system, stories, rc, story = 'brbf', '9', 'iv', 'max-max'

    for rc, system, stories in product(
        ('ii', 'iv'), ('smrf', 'scbf', 'brbf'), ('3', '6', '9')
    ):
        story = 'max-max'

        # Stratification criterion, only applicable when story='max-max'.
        # 'StoryDiffText': Difference in story where maxPID/maxRID occur.
        # 'PIDStoryText':  Story where max PID occurs.
        # 'RIDStoryText':  Story where max RID occurs.

        criterion = 'StoryDiffText'
        # criterion = 'PIDStoryText'
        # criterion = 'RIDStoryText'

        df_1 = pd.read_parquet('data/edp_results_0.parquet')
        df_2 = pd.read_parquet('data/edp_results_26plus.parquet')
        df = pd.concat((df_1, df_2))

        df.index = df.index.reorder_levels(
            ['system', 'stories', 'rc', 'dir', 'edp', 'hz', 'gm', 'loc']
        )
        df = df.sort_index()
        df_collapse = pd.read_parquet('data/pr_collapse.parquet')

        # Residual drift fragility curve (for probabilities)
        rid_capacity_delta = 0.01
        rid_capacity_beta = 0.3

        # Show probability of collapse
        show_collapse_probability = False

        # plotting parameters
        xmin, xmax = -0.002, 0.02  # rid
        ymin, ymax = -0.002, 0.065  # pid

        generate_plot(
            df,
            df_collapse,
            selected_model,
            system,
            stories,
            rc,
            story,
            criterion,
            rid_capacity_delta,
            rid_capacity_beta,
            xmin,
            xmax,
            ymin,
            ymax,
            show_collapse_probability,
        )


def get_parameter_table():

    story = 'max-max'
    df_1 = pd.read_parquet('data/edp_results_0.parquet')
    df_2 = pd.read_parquet('data/edp_results_26plus.parquet')
    df = pd.concat((df_1, df_2))
    df.index = df.index.reorder_levels(
        ['system', 'stories', 'rc', 'dir', 'edp', 'hz', 'gm', 'loc']
    )
    df = df.sort_index()

    parameter_data = defaultdict(lambda: defaultdict(list))
    median_data = defaultdict(lambda: defaultdict(list))
    for model, rc, system, stories in tqdm(
        list(
            product(
                ('trilinear',),
                ('ii', 'iv'),
                ('smrf', 'scbf', 'brbf'),
                ('3', '6', '9'),
            )
        )
    ):
        print()
        print(model, rc, system, stories)
        print()
        if model == 'bilinear':
            selected_model = models.Model_1_Weibull()
        elif model == 'trilinear':
            selected_model = models.Model_Weibull_Trilinear()
        else:
            raise ValueError('Invalid model: `{model}`.')

        pairs = get_pid_rid_pairs(df, system, stories, rc, story)
        pelicun_fitted_model = selected_model
        weibull_subset = pairs
        if model == 'bilinear':
            weibull_subset = weibull_subset[weibull_subset['PID'] > 0.02]
        pids = weibull_subset['PID'].values
        rids = weibull_subset['RID'].values
        pelicun_fitted_model.add_data(pids, rids)
        pelicun_fitted_model.censoring_limit = 0.00025
        pelicun_fitted_model.fit(method='mle')
        parameter_data[model][
            (system, stories, rc)
        ] = pelicun_fitted_model.parameters

        roll = models.Model()
        roll.add_data(pairs['PID'].values, pairs['RID'].values)
        roll.calculate_rolling_quantiles()

        model_pid = np.linspace(0.00, 0.06, 1000)
        model_rid_50 = pelicun_fitted_model.evaluate_inverse_cdf(0.50, model_pid)
        model_rid_20 = pelicun_fitted_model.evaluate_inverse_cdf(0.20, model_pid)
        model_rid_80 = pelicun_fitted_model.evaluate_inverse_cdf(0.80, model_pid)

        median_data[model][(system, stories, rc)] = model_rid_50

        c0 = '#7C296A'

        fig = make_subplots(
            rows=1,
            cols=1,
            shared_yaxes=True,
            subplot_titles=(
                f'{system}_{stories}_{rc} story {story} RID|PID distribution',
            ),
        )

        criterion = 'StoryDiffText'

        categories = pairs[criterion].unique()

        palette = {
            'Same story': "#7B9F35",
            '1--2 stories': "#226666",
            '>2 stories': "#AA3939",
            'First 2 stories': "#7B9F35",
            'First 3 stories': "#7B9F35",
            'Intermediate stories': "#226666",
            'Last story': "#226666",
        }

        for category in categories:

            custom_data = list(
                zip(
                    pairs[pairs[criterion] == category]['dir'],
                    pairs[pairs[criterion] == category]['hz'],
                    pairs[pairs[criterion] == category]['PIDStory'],
                    pairs[pairs[criterion] == category]['RIDStory'],
                )
            )
            hovertemplate = (
                'RID: %{x}<br>'
                'PID: %{y}<br>'
                'DIR: %{customdata[0]}<br>'
                'HZLVL: %{customdata[1]}<br>'
                'PID Story: %{customdata[2]}<br>'
                'RID Story: %{customdata[3]}<extra></extra>'
            )

            fig.add_trace(
                go.Scatter(
                    x=pairs[pairs[criterion] == category]['RID'],
                    y=pairs[pairs[criterion] == category]['PID'],
                    mode='markers',
                    marker=dict(
                        color=palette[category],
                    ),
                    name=category,
                    customdata=custom_data,
                    hovertemplate=hovertemplate,
                ),
                row=1,
                col=1,
            )

        empirical_data = [
            (roll.rolling_rid_50, roll.rolling_pid, 3.5, True),
            (roll.rolling_rid_20, roll.rolling_pid, 1.5, False),
            (roll.rolling_rid_80, roll.rolling_pid, 1.5, False),
        ]

        cw_data = [
            (model_rid_50, model_pid, 3.5, True),
            (model_rid_20, model_pid, 1.5, False),
            (model_rid_80, model_pid, 1.5, False),
        ]

        # pylint: disable=cell-var-from-loop
        def add_traces(data, color, name, legendgroup):
            for x, y, width, showlegend in data:
                fig.add_trace(
                    go.Scatter(
                        x=x,
                        y=y,
                        mode='lines',
                        line={'color': color, 'width': width},
                        name=name if showlegend else None,
                        showlegend=showlegend,
                        legendgroup=legendgroup,
                    ),
                    row=1,
                    col=1,
                )
        # pylint: enable=cell-var-from-loop

        add_traces(empirical_data, 'black', 'Empirical rolling quantiles', 'erq')
        add_traces(cw_data, c0, 'Conditional Weibull', 'cw')

        # Update layout to position the legend at the top
        fig.update_layout(
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
            ),
            height=900,
        )
        fig.update_xaxes(title_text="RID", row=1, col=1)
        fig.update_yaxes(title_text="PID", row=1, col=1)
        fig.show()

    # param_df_bilinear = pd.DataFrame(parameter_data['bilinear']).T.rename_axis(
    #     index=['system', 'stories', 'rc']
    # )
    param_df_trilinear = pd.DataFrame(parameter_data['trilinear']).T.rename_axis(
        index=['system', 'stories', 'rc'],
    )
    param_df_trilinear.columns = ['PID_0', 'lm_0', 'PID_1', 'lm_1', 'k_1', 'k_2']

    print(param_df_trilinear.to_latex())

    # back-calculate the medians
    all_series = []
    for system, stories, rc in tqdm(
        list(
            product(
                ('smrf', 'scbf', 'brbf'),
                ('3', '6', '9'),
                ('ii', 'iv'),
            )
        )
    ):

        row = (system, stories, rc)
        parameters = param_df_trilinear.loc[row, :]
        temp_model = models.Model_Weibull_Trilinear()
        temp_model.parameters = parameters.values
        model_pid = np.linspace(0.00, 0.06, 1000)
        lambda_vals, kappa_vals = temp_model.obtain_lambda_and_kappa(model_pid)
        medians = lambda_vals * np.log(2.00) ** (1.00 / kappa_vals)
        evaluated_medians = median_data['trilinear'][row]
        assert np.all(medians == evaluated_medians)
        all_series.append(pd.Series(medians, index=model_pid, name=row))
    df = pd.concat(all_series, axis=1)
    df.index.names = ['PID']
    df.columns.names = ['system', 'stories', 'rc']
    df.to_csv('doc/communications/median_curves.csv')

    fig, ax = plt.subplots()
    ax.plot(df.loc[:, ('scbf', '6', 'iv')])
    plt.show()

    # working on imposing constraints on the parameters

    filtered = param_df_trilinear
    # filtered= filtered.drop('smrf')

    def param_boxplot():
        for parameter in param_df_trilinear.columns:
            # parameter = 'lm_0'
            # Create subplots
            df = filtered.reset_index()
            fig = sp.make_subplots(
                rows=1,
                cols=3,
                subplot_titles=(
                    "Boxplot by System",
                    "Boxplot by Stories",
                    "Boxplot by RC",
                ),
            )

            # Boxplot by System
            for system in df['system'].unique():
                fig.add_trace(
                    go.Box(y=df[df['system'] == system][parameter], name=system),
                    row=1,
                    col=1,
                )

            # Boxplot by Stories
            for stories in df['stories'].unique():
                fig.add_trace(
                    go.Box(
                        y=df[df['stories'] == stories][parameter], name=str(stories)
                    ),
                    row=1,
                    col=2,
                )

            # Boxplot by RC
            for rc in df['rc'].unique():
                fig.add_trace(
                    go.Box(y=df[df['rc'] == rc][parameter], name=rc), row=1, col=3
                )

            # Update layout
            fig.update_layout(
                height=400,
                width=1000,
                title_text=(
                    f"`{parameter}` Boxplots Partitioned by System, Stories, and RC."
                ),
            )

            # Show the plot
            fig.show()

    pidrat = param_df_trilinear['PID_1'] / param_df_trilinear['PID_0']
    # smrf doesn't count
    pidrat.drop('smrf', inplace=True)


if __name__ == '__main__':
    main()
