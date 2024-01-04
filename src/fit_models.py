"""
Fit all models to the data and store the parameters.
"""

import pandas as pd
from itertools import product
import tqdm
from src.models import Model_1_Weibull
from src.handle_data import load_dataset
from src.handle_data import remove_collapse
from src.handle_data import only_drifts
from src.util import store_info


def get_all_cases():
    cases = []
    for sys, st, rc, dr in product(
        ('smrf', 'scbf', 'brbf'),  # system
        ('3', '6', '9'),  # number of stories
        ('ii', 'iv'),  # risk category
        ('1', '2'),  # direction (X, Y)
    ):
        for lv in range(1, int(st) + 1):
            lv = int(lv)
            cases.append((sys, st, rc, str(lv), dr))

    return cases


def obtain_weibull_parameters():
    df = only_drifts(remove_collapse(load_dataset()[0]))

    cases = get_all_cases()

    parameters = []
    loglikelihood = []

    for the_case in tqdm.tqdm(cases):
        case_df = df[the_case].dropna()
        rid_vals = case_df.dropna()["RID"].to_numpy().reshape(-1)
        pid_vals = case_df.dropna()["PID"].to_numpy().reshape(-1)

        model = Model_1_Weibull()
        model.add_data(pid_vals, rid_vals)
        model.censoring_limit = 0.0005
        model.fit(method='mle')
        loglikelihood.append(-model.fit_meta.fun)
        parameters.append(model.parameters)

    res = pd.DataFrame(
        parameters,
        index=pd.MultiIndex.from_tuples(
            cases, names=('system', 'stories', 'rc', 'story', 'direction')
        ),
        columns=['c_pid_0', 'c_lamda_slope', 'c_kapa'],
    )
    res['loglikelihood'] = loglikelihood
    res.sort_index(inplace=True)
    res.to_parquet(
        store_info(
            'results/parameters/weibull_bilinear/parameters.parquet',
            ['data/edp.parquet'],
        )
    )


def main():
    obtain_weibull_parameters()


if __name__ == '__main__':
    main()
