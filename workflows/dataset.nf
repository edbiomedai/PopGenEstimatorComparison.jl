include { MakeDatasetFromVariants } from '../subworkflows/dataset.nf'

workflow MAKE_DATASET {
    variant_list = Channel.value(file(params.VARIANTS_LIST, checkIfExists: true))
    MakeDatasetFromVariants(variant_list)
}