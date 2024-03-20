include { JuliaCmd; LongestPrefix } from '../modules/functions.nf'

process DensityEstimationInputs {
    publishDir "${params.OUTDIR}/density_estimation", mode: 'symlink'

    input:
        path dataset 
        path estimands
        
    output:
        path "conditional_density_*", emit: conditional_densities

    script:
        estimands_prefix = LongestPrefix(estimands)
        """
        ${JuliaCmd()} density-estimation-inputs ${dataset} ${estimands_prefix} \
            --output-prefix=conditional_density_
        """
}

process DensityEstimation {
    publishDir "${params.OUTDIR}/density_estimation/density_estimates", mode: 'symlink'

    input:
        path dataset 
        path density
        path estimators
        
    output:
        path outfile

    script:
        file_id = density.name[0..-6].split("_")[-1]
        outfile = "conditional_density_estimate_${file_id}.hdf5"
        """
        ${JuliaCmd()} density-estimation ${dataset} ${density} \
            --estimators=${estimators} \
            --output=${outfile} \
            --train-ratio=${params.TRAIN_RATIO} \
            --verbosity=${params.VERBOSITY}
        """
}

workflow DENSITY_ESTIMATION {
    dataset = Channel.value(file(params.DATASET, checkIfExists: true))
    estimands = Channel.fromPath(params.ESTIMANDS, checkIfExists: true)
    density_estimators = Channel.value(file(params.DENSITY_ESTIMATORS, checkIfExists: true))

    de_inputs = DensityEstimationInputs(dataset, estimands.collect())

    DensityEstimation(
        dataset,
        de_inputs.conditional_densities.flatten(),
        density_estimators
    )
}