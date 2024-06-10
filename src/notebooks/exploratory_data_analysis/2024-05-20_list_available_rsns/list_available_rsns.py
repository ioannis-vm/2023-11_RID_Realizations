# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%

# %% [markdown]
"""
# Find all RSNS that are already available in `data/`

`Mon May 20 12:32:19 PM PDT 2024`

"""

# %% [markdown]
"""
### Setup
"""

# %%
# Change directory to project's root, if needed
# (not needed when re-evaluating)
# pylint: disable=wrong-import-position
from pathlib import Path
import os

if Path(os.getcwd()).name != '2023-11_RID_Realizations':
    os.chdir('../../../../')


# %%
# Imports
from glob2 import glob

# %%
result_csvs = glob(
    'extra/structural_analysis/data/ground_motions/*/*/_SearchResults.csv'
)

all_rsns = []

for csvfile in result_csvs:
    with open(csvfile, 'r', encoding='utf-8') as f:
        contents = f.read()  # type: ignore

    contents = contents.split(' -- Summary of Metadata of Selected Records --')[
        1
    ]  # type: ignore
    contents = contents.split('\n\n')[0]  # type: ignore
    contents = contents.split('\n')[2:]  # type: ignore
    rsns = []
    for line in contents:
        rsns.append(int(line.split(',')[2].strip()))

    all_rsns.extend(rsns)

# %%
all_rsns.sort()
print(all_rsns)
