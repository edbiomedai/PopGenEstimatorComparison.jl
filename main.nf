#!/usr/bin/env nextflow
nextflow.enable.dsl = 2

params.OUTDIR = "${launchDir}/results"
params.VERBOSITY = 0
params.TL_SAVE_EVERY = 100
params.N_REPEATS = 100
params.RNGS = [0]
params.TRAIN_RATIO = 10
params.ESTIMATORS = "assets/estimators-configs/*"
params.N_FOR_TRUTH = 500000
params.DISTANCE_THRESHOLD = 1000000
params.GA_TRAIT_TABLE = "assets/Traits_Table_GeneATLAS.csv"

include { PERMUTATION_ESTIMATION } from './workflows/permutation.nf'
include { DENSITY_ESTIMATION } from './workflows/density_estimation.nf'

workflow {
    PERMUTATION_ESTIMATION()
    DENSITY_ESTIMATION()
}