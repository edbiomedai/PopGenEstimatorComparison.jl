include { JuliaCmd } from '../modules/functions.nf'
include { AggregateResults } from '../modules/aggregate.nf'

process PermutationEstimation {
    publishDir "${params.OUTDIR}/permutation_estimation", mode: 'symlink'

    input:
        path origin_dataset 
        tuple path(estimators), path(estimands), val(sample_size), val(rng)
        
    output:
        path out

    script:
        out = "permutation_results__${rng}__${sample_size}__${estimands.getBaseName()}__${estimators.getBaseName()}.hdf5"
        """
        mkdir workdir
        ${JuliaCmd()} estimation ${origin_dataset} ${estimands} ${estimators} \
            --sample-size=${sample_size} \
            --n-repeats=${params.N_REPEATS} \
            --out=${out} \
            --verbosity=${params.VERBOSITY} \
            --chunksize=${params.TL_SAVE_EVERY} \
            --rng=${rng} \
            --workdir=workdir
        """
}

workflow PERMUTATION_ESTIMATION {
    origin_dataset = Channel.value(file(params.DATASET, checkIfExists: true))

    estimators = Channel.fromPath(params.ESTIMATORS, checkIfExists: true)
    estimands = Channel.fromPath(params.ESTIMANDS, checkIfExists: true)
    sample_sizes = Channel.fromList(params.SAMPLE_SIZES)
    rngs = Channel.fromList(params.RNGS)
    combined = estimators.combine(estimands).combine(sample_sizes).combine(rngs)

    permutation_results = PermutationEstimation(origin_dataset, combined)
    AggregateResults(permutation_results.collect(), "permutation_results.hdf5")
}