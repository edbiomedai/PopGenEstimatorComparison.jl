process PermutationEstimation {
    input:
        path origin_dataset 
        tuple path(estimators), path(estimands), val(sample_size)
        
    // output:
    //     path 

    script:
        nrepeats = params.NREPEATS
        rng = params.RNG
        println(estimators)
}

workflow PERMUTATION_NULL_ESTIMATION {
    origin_dataset = Channel.value(file(params.DATASET))

    estimators = Channel.fromPath(params.ESTIMATORS)
    estimands = Channel.fromPath(params.ESTIMANDS)
    sample_sizes = Channel.fromList(params.SAMPLE_SIZES)
    combined = estimators.combine(estimands).combine(sample_sizes)

    PermutationEstimation(origin_dataset, combined)
}