# pandas-ts

## WIP exploratory repository for nested array representation of time series data

## Potential features

### Easy to implement

-  Replace `pack_dfs()`, `pack_lists()`, `pack_flat()` with a single function
- Zero-copy nesting of dataframes having pre-grouped index
- Reading parquet
- Reading CSV (textual formats)
- `.ts["field_name"] +=`, `/=`, etc
- Multiindex support
- String representation of a nested item (nested dataframe)

### Doable, but a lot of work

- Unpin pandas version. Currently we build on pandas `ArrowExtensionArray`, which is marked experimental. It would also improve performance.
- **Non-arrow types.** We actually can reimplement everything without pyarrow and nested arrays

### Tricky or impossible to implement

- **Performant dataframe representation of items as `pd.DataFrame`.** Currently it is ~10x-200x overhead versus `pyarrow`'s convertion to Python `dict`, ~20-50 μs per item.
- `df.eval()`, `df.query()` for dataframe with mixed "scalar" and "nested" columns.

### Scope is not clear

- Dask Dataframe integration
- Single series element assigment from dataframe doesn't wotk right now: `pd.Series.iloc[i] = pd.DataFrame(...)`

[![Template](https://img.shields.io/badge/Template-LINCC%20Frameworks%20Python%20Project%20Template-brightgreen)](https://lincc-ppt.readthedocs.io/en/latest/)

[![PyPI](https://img.shields.io/pypi/v/pandas-ts?color=blue&logo=pypi&logoColor=white)](https://pypi.org/project/pandas-ts/)
[![GitHub Workflow Status](https://img.shields.io/github/actions/workflow/status/lincc-frameworks/pandas-ts/smoke-test.yml)](https://github.com/lincc-frameworks/pandas-ts/actions/workflows/smoke-test.yml)
[![codecov](https://codecov.io/gh/lincc-frameworks/pandas-ts/branch/main/graph/badge.svg)](https://codecov.io/gh/lincc-frameworks/pandas-ts)
[![Read the Docs](https://img.shields.io/readthedocs/pandas-ts)](https://pandas-ts.readthedocs.io/)
[![benchmarks](https://img.shields.io/github/actions/workflow/status/lincc-frameworks/pandas-ts/asv-main.yml?label=benchmarks)](https://lincc-frameworks.github.io/pandas-ts/)

This project was automatically generated using the LINCC-Frameworks 
[python-project-template](https://github.com/lincc-frameworks/python-project-template).

A repository badge was added to show that this project uses the python-project-template, however it's up to
you whether or not you'd like to display it!

For more information about the project template see the 
[documentation](https://lincc-ppt.readthedocs.io/en/latest/).

## Dev Guide - Getting Started

Before installing any dependencies or writing code, it's a great idea to create a
virtual environment. LINCC-Frameworks engineers primarily use `conda` to manage virtual
environments. If you have conda installed locally, you can run the following to
create and activate a new environment.

```
>> conda create env -n <env_name> python=3.10
>> conda activate <env_name>
```

Once you have created a new environment, you can install this project for local
development using the following commands:

```
>> pip install -e .'[dev]'
>> pre-commit install
>> conda install pandoc
```

Notes:
1) The single quotes around `'[dev]'` may not be required for your operating system.
2) `pre-commit install` will initialize pre-commit for this local repository, so
   that a set of tests will be run prior to completing a local commit. For more
   information, see the Python Project Template documentation on 
   [pre-commit](https://lincc-ppt.readthedocs.io/en/latest/practices/precommit.html)
3) Install `pandoc` allows you to verify that automatic rendering of Jupyter notebooks
   into documentation for ReadTheDocs works as expected. For more information, see
   the Python Project Template documentation on
   [Sphinx and Python Notebooks](https://lincc-ppt.readthedocs.io/en/latest/practices/sphinx.html#python-notebooks)
