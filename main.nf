#!/usr/bin/env nextflow
nextflow.enable.dsl = 2

params.OUTDIR = "${launchDir}/results"
params.VERBOSITY = 0
params.TL_SAVE_EVERY = 100
params.N_REPEATS = 100
params.RNG = 0

include { PERMUTATION_NULL_ESTIMATION } from './workflows/permutation.nf'

workflow {
    
}