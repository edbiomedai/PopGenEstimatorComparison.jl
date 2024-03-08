# PopGenEstimatorComparison

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://olivierlabayle.github.io/PopGenEstimatorComparison.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://olivierlabayle.github.io/PopGenEstimatorComparison.jl/dev/)
[![Build Status](https://github.com/olivierlabayle/PopGenEstimatorComparison.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/olivierlabayle/PopGenEstimatorComparison.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/olivierlabayle/PopGenEstimatorComparison.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/olivierlabayle/PopGenEstimatorComparison.jl)
[![Code Style: Blue](https://img.shields.io/badge/code%20style-blue-4495d1.svg)](https://github.com/invenia/BlueStyle)

Double robust (DR) estimators have gained significant traction in the past years due to the theoretical benefits they provide. Many simulation studies have been conducted to analyse the performance of these estimators, especially in the context of model misspecification. However, the field of population genetics presents a set of unique and challenging features that haven't been extensively studied yet. In particular, modern biobanks such as the UK-Biobank offer a large sample size but small effect sizes and rare treatments and outcomes. The goal of this project is thus to provide a realistic simulation study, to evaluate the performance of semi-parametric DR estimators in population genetics. For these simulations to be as realistic as possible we base them on real world data from the UK-Biobank. We then fit data adaptive generative models from which ground truth effect sizes can be obtained and new data sampled from. Various DR estimators variations are evaluated based on these simulations.

## Simulation Study Workflow

The workflow can loosely be decomposed into four main steps:

1. Extract a template dataset from the UK-Biobank.
2. Fit generative models on the extracted dataset.
3. Sample from the generative models and run the estimators
4. Analyse the results

## Dependencies

There is no environment file at the moment, please install:

- Julia >= 1.10
- Nextflow >= 23.10.0
- Singularity >= 3.8.6 (Optional)