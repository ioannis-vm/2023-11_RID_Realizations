# "RID Realizations" project
Research analysis data for the *improved FEMA P-58 residual drift generation* project.

## Setting up the environment

```
$ conda create -n rid_prj -c conda-forge python=3.12 -y
$ conda activate rid_prj
$ python -m pip install -r requirements.txt
```

The interpreter is always assumed to be launched at the project root (the directory containing `src`, not `src` itself).

## Directory structure

`src/` contains the source code.

`src/validation` contains code that is not meant to reproduce the results of the study, but was utilized while developing the main code to take incremental steps towards a goal or to troubleshoot issues.

`data/` contains the time-history analysis data we use.

`results` contains analysis results, such as fit parameters or generated figures, and should not be version-controled.
We use the following directory structure: `results/{result-category}/{method}/{result-filename}`.
`{result-category}` can be any of `parameters`, `tables`, `figures`.
`{method}` corresponds to the fitting methods we examine.
`{result-filename}` is the file name for a type of result, and it can be the same for different methods.


`doc/` contains the files used to generate documents associated with the project, such as the poster presentation, manuscripts and any other derivative work. It also contaings an activity log.

## Data version control

`results` is not version controlled, but the contents are tracked with [DVC](https://dvc.org/).

## Data processing pipeline

This section will describe in what order to execute the code to reproduce all of the project's analysis results.
Currently all code is merely exploratory.

