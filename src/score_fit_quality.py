"""
Assign a numeric score to the fit qualtiy of each model

"""
from itertools import product
import pickle
import numpy as np
import scipy as sp
import pandas as pd
from src.handle_data import load_dataset
from src.handle_data import remove_collapse
from src.handle_data import only_drifts
import seaborn as sns
import matplotlib.pyplot as plt


def load_analysis_data() -> pd.DataFrame:
    """
    Reads the analysis result data in memory
    """
    data = only_drifts(remove_collapse(load_dataset()[0]))
    return data


def get_specific_case_results(
    all_data: pd.DataFrame, system: str, stories: str, rc: str, hazard_level: str
) -> pd.DataFrame:
    """
    Retrieves PID/RID pairs corresponding to a particular hazard level
    """
    out = (
        all_data.loc[hazard_level, (system, stories, rc)]
        .dropna(axis=1, how='all')  # type: ignore
        .dropna(axis=0, how='all')  # type: ignore
    )
    return out


def load_pretrained_models() -> dict[tuple[str, str], pd.Series]:
    """
    Reads the fitted models in memory
    """
    methods = (
        'weibull_bilinear',
        'gamma_bilinear',
    )
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

    return model_dfs


def get_cdfs(
    data: pd.DataFrame,
    model_dfs: dict[tuple[str, str], pd.Series],
    method: str,
    data_gathering_approach: str,
    system: str,
    stories: str,
    rc: str,
    hazard_level: str,
    location: str,
    direction: str,
) -> tuple[np.ndarray, np.ndarray]:
    num_realizations = 1000
    results = get_specific_case_results(data, system, stories, rc, hazard_level)
    loc_dir_results = results[location][direction]
    analysis_pid = loc_dir_results['PID'].values
    analysis_rid = loc_dir_results['RID'].values
    model = model_dfs[(method, data_gathering_approach)][
        (system, stories, rc, location, direction)
    ]
    pid_bootstrap_idx = np.random.choice(
        range(len(analysis_pid)), size=num_realizations
    )
    simulated_pid = analysis_pid[pid_bootstrap_idx]
    simulated_rid = model.generate_rid_samples(simulated_pid)

    return simulated_rid, analysis_rid


def get_p_value(simulated_rid: np.ndarray, analysis_rid: np.ndarray) -> float:
    # simulated_rid, analysis_rid = get_cdfs(hazard_level)
    return sp.stats.kstest(simulated_rid, analysis_rid).pvalue


def plot_cdfs(simulated_rid, analysis_rid):
    fig, ax = plt.subplots()
    sns.ecdfplot(analysis_rid, ax=ax)
    sns.ecdfplot(simulated_rid, ax=ax)
    ax.axvline(x=0.01)
    ax.set(xlim=(-0.005, 0.105))
    ax.set(ylim=(-0.05, 1.05))
    plt.show()


def probability_of_excessive_drift(rids: np.ndarray) -> float:
    num_realizations = 1000
    theta = 0.01
    beta = 0.30
    if len(rids) < num_realizations:
        bootstrap_idx = np.random.choice(range(len(rids)), size=num_realizations)
        rids = rids[bootstrap_idx]
    sampler = sp.stats.qmc.LatinHypercube(d=1)
    uniform_sample = sampler.random(n=num_realizations).reshape(-1)
    capacities = np.exp(
        sp.stats.norm.ppf(uniform_sample, loc=np.log(theta), scale=beta)
    )
    probability = np.sum(rids > capacities, dtype=float) / num_realizations
    return probability


def get_probabilities(
    simulated_rid: np.ndarray, analysis_rid: np.ndarray
) -> tuple[float, float]:
    p_simulated = probability_of_excessive_drift(simulated_rid)
    p_analysis = probability_of_excessive_drift(analysis_rid)
    return p_analysis, p_simulated


data = load_analysis_data()
model_dfs = load_pretrained_models()

result_data = {}
for index in product(
    ('gamma_bilinear', 'weibull_bilinear'),
    ('separate_directions',),
    ('smrf', 'scbf', 'brbf'),
    ('3', '6', '9'),
    ('ii', 'iv'),
    [f'{i+1}' for i in range(9)],
    ('1', '2'),
    [f'{i+1}' for i in range(8)],
):
    (
        method,
        data_gathering_approach,
        system,
        stories,
        rc,
        location,
        direction,
        hazard_level,
    ) = index

    location_int = int(location)
    stories_int = int(stories)
    if location > stories:
        continue

    simulated_rid, analysis_rid = get_cdfs(
        data,
        model_dfs,
        method,
        data_gathering_approach,
        system,
        stories,
        rc,
        hazard_level,
        location,
        direction,
    )
    p_value = get_p_value(simulated_rid, analysis_rid)
    probabilities = get_probabilities(simulated_rid, analysis_rid)
    result_data[index] = {
        'p_value': p_value,
        'p1': probabilities[0],
        'p2': probabilities[1],
    }

result_df = pd.DataFrame(result_data).T
result_df.index.names = (
    'method',
    'data_gathering_approach',
    'system',
    'stories',
    'rc',
    'location',
    'direction',
    'hazard_level',
)

# Determining the best performing distribution in terms of the KS test results

# remove cases with negligible probability of excessive drift

stacked_methods = result_df.unstack(0)
filtered_df = stacked_methods[stacked_methods['p1'] > 0.01].dropna(
    axis=0, how='all'
)['p_value']

fig, ax = plt.subplots()
sns.histplot(filtered_df, kde=True, fill=False, ax=ax)
sns.despine(fig)
plt.show()


# Determining the best performing distribution in terms of the
# difference in probabilyt of excessive drift
result_df['p_diff'] = (result_df['p2'] - result_df['p1'])
filtered_df = result_df['p_diff'].unstack(0)
filtered_df[filtered_df.index.get_level_values('system') == 'brbf'].describe()
# fig, ax = plt.subplots()
# sns.histplot(filtered_df, kde=True, fill=False, ax=ax)
# sns.despine(fig)
# plt.show()

# if __name__ == '__main__':
#     main()
