include { MakeDatasetFromVariants } from '../subworkflows/dataset.nf'
include { PermutationSimulation } from '../subworkflows/permutation_simulation.nf'

workflow PERMUTATION_SIMULATION {
    variant_list = Channel.value(file(params.VARIANTS_LIST, checkIfExists: true))
    dataset = MakeDatasetFromVariants(variant_list)
    PermutationSimulation(dataset)
}