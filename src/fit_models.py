"""
Fit all models to the data and store the parameters.
"""

from itertools import product
import pickle
import pandas as pd
import tqdm
from src import models
from src.handle_data import load_dataset
from src.handle_data import remove_collapse
from src.handle_data import only_drifts
from src.util import store_info


def get_all_cases(data_gathering_approach: str) -> list[tuple[str, ...]]:
    cases: list[tuple[str, ...]] = []
    if data_gathering_approach == 'separate_directions':
        for sys, st, rc, dr in product(
            ('smrf', 'scbf', 'brbf'),  # system
            ('3', '6', '9'),  # number of stories
            ('ii', 'iv'),  # risk category
            ('x', 'y'),  # direction (X, Y)
        ):
            for lv in range(1, int(st) + 1):
                lv = int(lv)
                cases.append((sys, st, rc, str(lv), dr))
    elif data_gathering_approach == 'bundled_directions':

        for sys, st, rc in product(
            ('smrf', 'scbf', 'brbf'),  # system
            ('3', '6', '9'),  # number of stories
            ('ii', 'iv'),  # risk category
        ):
            for lv in range(1, int(st) + 1):
                lv = int(lv)
                cases.append((sys, st, rc, str(lv)))
    else:
        raise ValueError(
            f'Invalid data_gathering_approach: {data_gathering_approach}'
        )

    return cases


def obtain_parameters(
    method: str,
    data_gathering_approach: str,
) -> None:
    df = only_drifts(remove_collapse(load_dataset()[0]))
    cases = get_all_cases(data_gathering_approach)
    parameters = []
    loglikelihood = []
    model_objects = {}

    model_classes = {
        'weibull_bilinear': models.Model_Bilinear_Weibull,
        'gamma_bilinear': models.Model_Bilinear_Gamma,
    }
    for the_case in cases:
        case_df: pd.DataFrame = df[the_case].dropna()  # type: ignore
        if data_gathering_approach == 'bundled_directions':
            stack = case_df.stack(level=0, future_stack=True)
            assert isinstance(stack, pd.DataFrame)
            case_df = stack
        rid_vals = case_df.dropna()["RID"].to_numpy().reshape(-1)
        pid_vals = case_df.dropna()["PID"].to_numpy().reshape(-1)

        model = model_classes[method]()
        model.add_data(pid_vals, rid_vals)
        model.censoring_limit = 0.0025
        model.fit(method='mle')
        loglikelihood.append(-model.fit_meta.fun)
        parameters.append(model.parameters)
        model_objects[the_case] = model

    if data_gathering_approach == 'bundled_directions':
        multiindex = pd.MultiIndex.from_tuples(
            cases, names=('system', 'stories', 'rc', 'story')
        )
    else:
        multiindex = pd.MultiIndex.from_tuples(
            cases, names=('system', 'stories', 'rc', 'story', 'direction')
        )

    res = pd.DataFrame(
        parameters,
        index=multiindex,
        columns=['c_pid_0', 'c_lamda_slope', 'c_kapa'],
    )
    res['loglikelihood'] = loglikelihood
    res.sort_index(inplace=True)
    res.to_parquet(
        store_info(
            f'results/parameters/{data_gathering_approach}/{method}/parameters.parquet',
            ['data/edp.parquet'],
        )
    )
    with open(
        store_info(
            f'results/parameters/{data_gathering_approach}/{method}/models.pickle',
            ['data/edp.parquet'],
        ),
        'wb',
    ) as f:
        pickle.dump(model_objects, f)


def main() -> None:
    for method, data_gathering_approach in tqdm.tqdm(
        list(
            product(
                (
                    'weibull_bilinear',
                    'gamma_bilinear',
                ),
                ('separate_directions', 'bundled_directions'),
            )
        )
    ):
        obtain_parameters(method, data_gathering_approach)


if __name__ == '__main__':
    main()
