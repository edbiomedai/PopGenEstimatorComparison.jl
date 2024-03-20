#!/usr/bin/env nextflow
nextflow.enable.dsl = 2

params.OUTDIR = "${launchDir}/results"
params.VERBOSITY = 0
params.TL_SAVE_EVERY = 100
params.N_REPEATS = 100
params.RNGS = [0]

include { PERMUTATION_ESTIMATION } from './workflows/permutation.nf'
include { DENSITY_ESTIMATION } from './workflows/density_estimation.nf'

workflow {}