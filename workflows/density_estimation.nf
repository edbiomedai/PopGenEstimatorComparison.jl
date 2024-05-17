include { JuliaCmd; LongestPrefix } from '../modules/functions.nf'
include { AggregateResults } from '../modules/aggregate.nf'
include { DensityEstimation } from '../modules/density_estimation.nf'

process Analyse {
    label 'bigmem'
    publishDir "${params.OUTDIR}/density_estimation", mode: 'symlink'

    input:
        path results_file
        path estimands_files
        path dataset_file
        path density_estimates_files
        
    output:
        path "analysis/analysis1D/summary_stats.hdf5"

    script:
        estimands_prefix = LongestPrefix(estimands_files)
        density_estimates_prefix = LongestPrefix(density_estimates_files)
        """
        ${JuliaCmd()} analyse \
            ${results_file} \
            ${estimands_prefix} \
            --out-dir=analysis \
            --n=${params.N_FOR_TRUTH} \
            --dataset-file=${dataset_file} \
            --density-estimates-prefix=${density_estimates_prefix}
        """
}

process DensityEstimationInputs {
    publishDir "${params.OUTDIR}/density_estimation/conditional_densities", mode: 'symlink', pattern: "*.json"
    publishDir "${params.OUTDIR}/density_estimation/estimands", mode: 'symlink', pattern: "*.jls"

    input:
        path dataset 
        path estimands
        
    output:
        path "*.json", emit: conditional_densities
        path "*.jls", emit: estimands

    script:
        estimands_prefix = LongestPrefix(estimands)
        """
        ${JuliaCmd()} density-estimation-inputs ${dataset} ${estimands_prefix} \
            --batchsize=${params.BATCH_SIZE} \
            --output-prefix=de_
        """
}

process EstimationFromDensityEstimates {
    label 'bigmem'
    publishDir "${params.OUTDIR}/estimation_from_de", mode: 'symlink'

    input:
        path origin_dataset 
        tuple path(estimators), path(density_estimates), path(estimands), val(sample_size), val(rng)
        
    output:
        path out

    script:
        out = "results__${rng}__${sample_size}__${estimands.getBaseName()}__${estimators.getBaseName()}.hdf5"
        density_estimate_prefix = LongestPrefix(density_estimates)
        sample_size_option = sample_size != -1 ? "--sample-size=${sample_size}" : ""
        """
        mkdir workdir
        ${JuliaCmd()} estimation ${origin_dataset} ${estimands} ${estimators} \
            --density-estimates-prefix=${density_estimate_prefix} \
            ${sample_size_option} \
            --n-repeats=${params.N_REPEATS} \
            --out=${out} \
            --verbosity=${params.VERBOSITY} \
            --chunksize=${params.TL_SAVE_EVERY} \
            --rng=${rng} \
            --workdir=workdir
        """
}

workflow DENSITY_ESTIMATION {
    dataset = Channel.value(file(params.DATASET, checkIfExists: true))
    estimands = Channel.fromPath(params.ESTIMANDS, checkIfExists: true)
    estimators = Channel.fromPath(params.ESTIMATORS, checkIfExists: true)
    sample_sizes = Channel.fromList(params.SAMPLE_SIZES)
    rngs = Channel.fromList(params.RNGS)
    
    // Density Estimation Inputs
    de_inputs = DensityEstimationInputs(dataset, estimands.collect())

    // Density Estimation
    density_estimates = DensityEstimation(
        dataset,
        de_inputs.conditional_densities.flatten(),
    )
    
    estimands_by_group = de_inputs.estimands
        .flatten()
        .map { [it.getName().split("_")[2], it] }
        
    grouped_density_estimates = density_estimates
        .map { [it.getName().split("_")[2], it] }
        .groupTuple()

    densities_with_estimand = grouped_density_estimates
        .cross(estimands_by_group)
        .map { [it[0][1], it[1][1]] }

    combined = estimators.combine(densities_with_estimand).combine(sample_sizes).combine(rngs)
    
    // Estimation
    estimates = EstimationFromDensityEstimates(
        dataset,
        combined
    )

    // Aggregation of Estimation Results
    AggregateResults(estimates.collect(), "from_densities_results.hdf5")
    
    // Analysis
    Analyse(
        AggregateResults.out, 
        estimands.collect(),
        dataset,
        density_estimates.collect()
    )
}