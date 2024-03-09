include { JuliaCmd } from '../modules/functions.nf'

process PermutationEstimation {
    publishDir "${params.OUTDIR}/permutation_estimation", mode: 'symlink'

    input:
        path origin_dataset 
        tuple path(estimators), path(estimands), val(sample_size)
        
    output:
        path out

    script:
        out = "permutation_results__${sample_size}__${estimands.getBaseName()}__${estimators.getBaseName()}.hdf5"
        """
        ${JuliaCmd()} permutation-estimation ${origin_dataset} ${estimands} ${estimators} \
            --sample-size=${sample_size} \
            --n-repeats=${params.N_REPEATS} \
            --out=${out} \
            --verbosity=${params.VERBOSITY} \
            --chunksize=${params.TL_SAVE_EVERY} \
            --rng=${params.RNG}
        """
}

process AggregateResults {
    publishDir "${params.OUTDIR}", mode: 'symlink'

    input:
        path results 
        
    output:
        path out

    script:
        out = "permutation_results.hdf5"
        """
        ${JuliaCmd()} aggregate permutation_results ${out}
        """

}

workflow PERMUTATION_NULL_ESTIMATION {
    origin_dataset = Channel.value(file(params.DATASET))

    estimators = Channel.fromPath(params.ESTIMATORS)
    estimands = Channel.fromPath(params.ESTIMANDS)
    sample_sizes = Channel.fromList(params.SAMPLE_SIZES)
    combined = estimators.combine(estimands).combine(sample_sizes)

    permutation_results = PermutationEstimation(origin_dataset, combined)
    AggregateResults(permutation_results.collect())
}