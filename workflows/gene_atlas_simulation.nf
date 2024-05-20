include { MAKE_DATASET } from './workflows/dataset.nf'
include { GeneATLASSimulation } from './subworkflows/gene_atlas_simulation.nf'

workflow GENE_ATLAS_SIMULATION {
    dataset = MAKE_DATASET()
    GeneATLASSimulation(dataset)
}