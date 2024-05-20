include { DensityEstimation } from '../modules/density_estimation.nf'
include { AggregateResults } from '../modules/aggregate.nf'
include { MakeDatasetFromVariants } from './dataset.nf'
include { JuliaCmd; LongestPrefix } from '../modules/functions.nf'

process GeneAtlasSimulationInputs {
    label 'bigmem'
    publishDir "${params.OUTDIR}/gene_atlas_simulation/estimands", mode: 'symlink', pattern: "*.jls"
    publishDir "${params.OUTDIR}/gene_atlas_simulation/conditional_densities", mode: 'symlink', pattern: "*.json"
    publishDir "${params.OUTDIR}/gene_atlas_simulation", mode: 'symlink', pattern: "*.txt"

    input:
        path estimands_files
        path ga_trait_table
        
    output:
        path "ga_sim_input_variants.txt", emit: variants
        path "ga_sim_input_conditional_density*", emit: conditional_densities
        path "ga_sim_input_estimand*", emit: estimands

    script:
        estimands_prefix = LongestPrefix(estimands_files)
        """
        ${JuliaCmd()} simulation-inputs-from-ga \
            ${estimands_prefix} \
            --ga-download-dir=gene_atlas_data \
            --ga-trait-table=${ga_trait_table} \
            --remove-ga-data=true \
            --maf-threshold=${params.GA_MAF_THRESHOLD} \
            --pvalue-threshold=${params.GA_PVAL_THRESHOLD} \
            --distance-threshold=${params.GA_DISTANCE_THRESHOLD} \
            --output-prefix=ga_sim_input \
            --batchsize=${params.BATCH_SIZE} \
            --max-variants=${params.MAX_VARIANTS}
        """
}

process EstimationFromDensityEstimates {
    label 'bigmem'
    publishDir "${params.OUTDIR}/gene_atlas_simulation/estimation_from_de", mode: 'symlink'

    input:
        path origin_dataset
        path density_estimates
        tuple path(estimators), path(estimands), val(sample_size), val(rng)
        
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

workflow GeneATLASSimulation {
    // General Simulation Input Grid
    estimands_files = Channel.fromPath(params.ESTIMANDS, checkIfExists: true)
    estimators = Channel.fromPath(params.ESTIMATORS, checkIfExists: true)
    sample_sizes = Channel.fromList(params.SAMPLE_SIZES)
    rngs = Channel.fromList(params.RNGS)

    // Other params
    ga_trait_table = Channel.value(file(params.GA_TRAIT_TABLE, checkIfExists: true))

    // Simulation Inputs
    simulation_inputs = GeneAtlasSimulationInputs(
        estimands_files.collect(),
        ga_trait_table
    )
    // Make Dataset
    dataset = MakeDatasetFromVariants(simulation_inputs.variants)

    // Density Estimation
    density_estimates = DensityEstimation(
        dataset,
        simulation_inputs.conditional_densities.flatten(),
    ).collect()

    estimands = simulation_inputs.estimands.flatten()

    combined = estimators.combine(estimands).combine(sample_sizes).combine(rngs)
    
    // Estimation
    estimates = EstimationFromDensityEstimates(
        dataset,
        density_estimates,
        combined
    )

    // Aggregation of Estimation Results
    AggregateResults(estimates.collect(), "gene_atlas_simulation_results.hdf5")
}