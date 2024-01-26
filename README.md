# "RID Realizations" project
Research analysis data for the *improved FEMA P-58 residual drift generation* project.

## Setting up the environment

```
# using venv
$ python -m venv {path/to/virtual_environment}
$ source {path/to/virtual_environment}/bin/activate
$ python -m pip install -r requirements.txt

# using conda
$ conda create -n rid_prj -c conda-forge python=3.12 -y
$ conda activate rid_prj
$ python -m pip install -r requirements.txt
```
You would then have to configure your editor to use the interpreter from the newly created environment. [(Emacs)](https://github.com/jorgenschaefer/pyvenv)
[(pycharm)](https://www.jetbrains.com/help/pycharm/creating-virtual-environment.html) [(spyder)](https://docs.spyder-ide.org/current/faq.html?highlight=venv#using-existing-environment)  
The interpreter is always assumed to be launched at the project root (the directory containing `src`, not `src` itself).

Tearing down the enviornment, if needed:
```
# using venv
$ rm -r {path/to/virtual_environment}

# using conda
conda remove -n rid_prj --all
```

## Directory structure

`src/` contains the source code.

`src/validation` contains code that is not meant to reproduce the results of the study, but was utilized while developing the main code to take incremental steps towards a goal or to troubleshoot issues.

`data/` contains the time-history analysis data we use.

`results` contains analysis results, such as fit parameters or generated figures, and should not be version-controled.  
We use the following directory structure: `results/{result-category}/{data_gathering_approach}/{method}/{result-filename}`.  
`{result-category}` can be any of `parameters`, `tables`, `figures`.  
{data_gathering_approach} can be any of `separate_directions`, `bundled_directions`.  
`{method}` corresponds to the fitting methods we examine.  
`{result-filename}` is the file name for a type of result, and it can be the same for different methods.  


`doc/` contains the files used to generate documents associated with the project, such as the poster presentation, manuscripts and any other derivative work.

## Data version control

`data/` and `results/` are not version controlled, but the contents are tracked with [DVC](https://dvc.org/).

After cloning the repository and setting up the environment, issue the following command to pull the contents:
```
$ dvc pull
```

After making changes, they should be added with DVC and then committed with git.
```

# changes in data/
$ dvc add data

# changes in results/
$ dvc add results

$ dvc push
$ git add {changed-dvc-files}
$ git commit -m 'DVC - update results'

```

## Data processing pipeline

This section will describe in what order to execute the code to reproduce all of the project's analysis results.
Note that the results should already be available using DVC.

Set the `PYTHONPATH` variable

```
export PYTHONPATH=$PYTHONPATH:$(pwd)
```

1. Fit all the models. Creates `parameters.parquet` and `models.picle` files in `results/parameters/{method}/`.
```
$ python src/fit_models.py
```

2. Create plots that help assess the quality of the fit. Creates `fit_{system}_{stories}_{rc}.pdf` files in `results/figures/{method}/`.
```
$ python src/plot_fit.py
```
