#!/usr/bin/env nextflow
nextflow.enable.dsl = 2

params.OUTDIR = "${launchDir}/results"
params.VERBOSITY = 0
params.TL_SAVE_EVERY = 100
params.N_REPEATS = 100
params.RNGS = [0]
params.TRAIN_RATIO = 10
params.ESTIMATORS = "${workflow.projectDir}assets/estimators-configs/*"
params.N_FOR_TRUTH = 500000
params.GA_DISTANCE_THRESHOLD = 1000000
params.GA_MAF_THRESHOLD = 0.01
params.GA_PVAL_THRESHOLD = 1e-5
params.GA_TRAIT_TABLE = "${workflow.projectDir}/assets/Traits_Table_GeneATLAS.csv"
params.UKB_WITHDRAWAL_LIST = "${workflow.projectDir}/assets/ukb_withdrawal_list.txt"
params.UKB_CONFIG = "${workflow.projectDir}/assets/ukbconfig.yaml"
params.FLASHPCA_EXCLUSION_REGIONS = "${projectDir}/assets/exclusion_regions_hg19.txt"
params.LD_BLOCKS = "${projectDir}/assets/NO_LD_BLOCKS"

include { PERMUTATION_ESTIMATION } from './workflows/permutation.nf'
include { DENSITY_ESTIMATION } from './workflows/density_estimation.nf'
include { GENE_ATLAS_SIMULATION } from './workflows/gene_atlas_simulation.nf'

workflow {
    PERMUTATION_ESTIMATION()
    DENSITY_ESTIMATION()
    // GENE_ATLAS_SIMULATION()
}