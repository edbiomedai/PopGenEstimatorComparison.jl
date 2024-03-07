#!/usr/bin/env nextflow
nextflow.enable.dsl = 2

include { PERMUTATION_NULL_ESTIMATION } from './workflows/permutation.nf'

workflow {
    
}