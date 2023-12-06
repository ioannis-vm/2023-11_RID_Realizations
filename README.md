# 2023-11_RID_Realizations
Research analysis data for the 2023/11 RID project.

## Setting up the environment

```
$ conda create -n rid_prj -c conda-forge python=3.12 -y
$ conda activate rid_prj
$ python -m pip install -r requirements.txt
```

The interpreter is always assumed to be launched at the project root (the directory containing `src`, not `src` itself).

### Directory structure

`src/` contains the source code.

`data/` contains the time-history analysis data we use.

`exploratory_data_analysis/` contains scripts we use on a one-time basis to explore the data and test out ideas.
There is no need to keep this directory particularly organized or update the scripts when we make breaking changes in `src`.
We simply keep them for reference, or in case we need to copy over and modify pieces of code in the future.
The filenames should start with a YYYYMMDD tag to make it easier to find things.

`output_data/` contains analysis outputs, such as fit parameters or generated figures, and should not be version-controled.

`doc/` contains the files used to generate documents associated with the project, such as the poster presentation, manuscripts and any other derivative work.

