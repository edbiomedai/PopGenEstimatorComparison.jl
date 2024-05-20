include { JuliaCmd; LongestPrefix } from '../modules/functions.nf'
include { AggregateResults } from '../modules/aggregate.nf'

process Analyse {
    label 'bigmem'
    publishDir "${params.OUTDIR}/permutation_estimation", mode: 'symlink'

    input:
        path results_file
        path estimands_files 
        
    output:
        path "analysis/analysis1D/summary_stats.hdf5"

    script:
        estimands_prefix = LongestPrefix(estimands_files)
        """
        ${JuliaCmd()} analyse \
            ${results_file} \
            ${estimands_prefix} \
            --out-dir=analysis
        """
}

process PermutationEstimation {
    label 'bigmem'
    publishDir "${params.OUTDIR}/permutation_estimation", mode: 'symlink'

    input:
        path origin_dataset 
        tuple path(estimators), path(estimands), val(sample_size), val(rng)
        
    output:
        path out

    script:
        out = "results__${rng}__${sample_size}__${estimands.getBaseName()}__${estimators.getBaseName()}.hdf5"
        sample_size_option = sample_size != -1 ? "--sample-size=${sample_size}" : ""
        """
        mkdir workdir
        ${JuliaCmd()} estimation ${origin_dataset} ${estimands} ${estimators} \
            ${sample_size_option} \
            --n-repeats=${params.N_REPEATS} \
            --out=${out} \
            --verbosity=${params.VERBOSITY} \
            --chunksize=${params.TL_SAVE_EVERY} \
            --rng=${rng} \
            --workdir=workdir
        """
}

workflow PermutationSimulation {
    take:
        dataset

    main:
        estimators = Channel.fromPath(params.ESTIMATORS, checkIfExists: true)
        estimands = Channel.fromPath(params.ESTIMANDS, checkIfExists: true)
        sample_sizes = Channel.fromList(params.SAMPLE_SIZES)
        rngs = Channel.fromList(params.RNGS)
        combined = estimators.combine(estimands).combine(sample_sizes).combine(rngs)

        // Estimation
        permutation_results = PermutationEstimation(dataset, combined)

        // Aggregation of Estimation Results
        AggregateResults(permutation_results.collect(), "permutation_simulation_results.hdf5")
    
        // Analysis
        Analyse(AggregateResults.out, estimands.collect())
}