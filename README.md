# PopGenEstimatorComparison

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://olivierlabayle.github.io/PopGenEstimatorComparison.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://olivierlabayle.github.io/PopGenEstimatorComparison.jl/dev/)
[![Build Status](https://github.com/olivierlabayle/PopGenEstimatorComparison.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/olivierlabayle/PopGenEstimatorComparison.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/olivierlabayle/PopGenEstimatorComparison.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/olivierlabayle/PopGenEstimatorComparison.jl)
[![Code Style: Blue](https://img.shields.io/badge/code%20style-blue-4495d1.svg)](https://github.com/invenia/BlueStyle)

Double robust (DR) estimators have gained significant traction in the past years due to the theoretical benefits they provide. Many simulation studies have been conducted to analyse the performance of these estimators, especially in the context of model misspecification. However, the field of population genetics presents a set of unique and challenging features that haven't been extensively studied yet. In particular, modern biobanks such as the UK-Biobank offer a large sample size but small effect sizes and rare treatments and outcomes. The goal of this project is thus to provide a realistic simulation study, to evaluate the performance of semi-parametric DR estimators in population genetics. For these simulations to be as realistic as possible we base them on real world data from the UK-Biobank. We then fit data adaptive generative models from which ground truth effect sizes can be obtained and new data sampled from. Various DR estimators variations are evaluated based on these simulations.

## What's in it ?

- A Nextflow pipeline to run simulation studies using Semi-Parametric Efficient Estimation.
- A specific run configuration (`dvc.yaml`) for a simulation study on gene-phenotype association studies based on the UK-Biobank.

## How to run the pipeline ?

### Dependencies

There is no environment file at the moment, please install:

- Julia >= 1.10
- Nextflow >= 23.10.0
- Singularity >= 3.8.6 (Optional)

### Running the pipeline

Assuming you have a `nextlow.config` file (parameters described below) in your current directory, you can run:

```bash
nextflow run https://github.com/edbiomedai/PopGenEstimatorComparison.jl -r COMMIT_ID -profile MY_PROFILE -resume
```

- `COMMIT_ID` is the pipeline version you want to point to.
- `MY_PROFILE` depends on the HPC you are running with and is simply `eddie` for University of Edinburgh users.

### Pipeline Description and Setup

The simulations provided by this pipeline are realistic in the sense that they are based on real-world data. In order to run a pipeline you will thus need to provide a dataset.

DATASET = "Path to Arrow dataset."

The goal of the simulation is to understand how well some quantities can be estimated using this dataset and semi-parametric estimators. This quantities are refered to as estimands and can be provided by one or multiples files in .jls (Julia Serialization format) or YAML format (see [TMLE.jl](https://targene.github.io/TMLE.jl/stable/) for a programmatic way to define them).

ESTIMANDS = "Path to Estimands"

Because we don't know the data generating process associated with our original dataset, we need to resort to simulations, there are two main ways provided in this pipeline:

- `PERMUTATION_ESTIMATION`: In this case, each column of the dataset is sampled with replacement independently, resulting in a theoretical 0 true effect for any estimand.
- `DENSITY_ESTIMATION`: In this case, the densities are learnt from the dataset using a Sieve Neural Network Estimator and new data is generated from these densities. This process is controled by `DENSITY_ESTIMATORS` which is a Julai file defaulting to "assets/density_estimators.jl". Do not change it unless you know what you are doing.

You can investigate how sample size influences the quality of estimators using:

SAMPLE_SIZES = [100, 200]

Finaly, the list of estimators under study can be provided in one or multiple files, this default to:

ESTIMATORS = "assets/estimators-configs/*"
    
Then, the pipeline runs a simulation study by bootstraping the estimation process. That is, a new simulated dataset is resampled a number of times according to the `PERMUTATION_ESTIMATION` or `DENSITY_ESTIMATION` strategy. The number of bootstrap samples is determined by the 2 following parameters:

N_REPEATS = 100

RNGS = [0, 1]

The number of bootstrap samples is `N_REPEATS*size(RNGS)` (in this case: 200). This is because `N_REPEATS` controls the number of times the dataset is resampled within the same Nextflow process while `RNGS` control the random seed of the process. The reason why it is designed that way is to maximise HPC platforms usage.  In this simple example, one process will run 100 repeats with a random seed of 0 and an other one with a random seed of 1.

Finally, results are aggregated in two steps:

1. Matching (by estimator and sample size) estimates are grouped in the same dataframes and saved in either:
    - results/from_densities_results.hdf5
    - results/permutation_results.hdf5
2. Summary statistics are computed across bootstrap samples and saved in either:
    - results/permutation_estimation/analysis/analysis1D/summary_stats.hdf5
    - results/density_estimation/analysis/analysis1D/summary_stats.hdf5

Those hdf5 files require a programming language (e.g. in Julia: JLD2) to be further analysed.

#### Misc Parameters

Changing the output directory:
- OUTDIR = "${launchDir}/results"

Changing how many estimands are estimated in a single process:
- BATCH_SIZE = 1

Changing the density estimation sample split ratio:
- TRAIN_RATIO = 10

Changing how many samples are used to compute the ground truth from the density estimates:
- N_FOR_TRUTH = 500000

Changing the verbosity level of Nextflow processes that allow it:
- VERBOSITY = 0

Changing how frequently data is saved when performing Targeted Estimation (to save memory usage):
- TL_SAVE_EVERY = 100

## The UK-Biobank Simulation Study

The workflow can loosely be decomposed into four main steps:

1. Extract a template dataset from the UK-Biobank.
2. Fit generative models on the extracted dataset.
3. Sample from the generative models and run the estimators
4. Analyse the results


## Density Estimators API

In order to be compatible with the model selection procedure, the density estimators must implement the following methods:

- A constructor taking two inputs (X, y) where X is a table and y a vector such that the returned estimator is compatible with (X, y).
- A `train!(estimator, X, y; verbosity)` method for fitting, returning the estimator itself.
- An `evaluation_metrics(estimator, X, y)` method returning the evaluation metrics as a NamedTuple.
- A sampling `sample_from()` method
