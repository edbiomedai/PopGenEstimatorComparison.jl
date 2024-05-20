include { MAKE_DATASET } from './workflows/dataset.nf'
include { PermutationSimulation } from './workflows/permutation_simulation.nf'

workflow PERMUTATION_SIMULATION {
    dataset = MAKE_DATASET()
    PermutationSimulation(dataset)
}