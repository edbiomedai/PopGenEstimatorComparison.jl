include { JuliaCmd; LongestPrefix } from '../modules/functions.nf'

process DensityEstimationInputs {
    publishDir "${params.OUTDIR}/density_estimation", mode: 'symlink'

    input:
        path dataset 
        path estimands
        
    output:
        path "conditional_densities_variables.json"

    script:
        estimands_prefix = LongestPrefix(estimands)
        """
        ${JuliaCmd()} density-estimation-inputs ${dataset} ${estimands_prefix}
        """
}

workflow DENSITY_ESTIMATION {
    dataset = Channel.value(file(params.DATASET, checkIfExists: true))
    estimands = Channel.fromPath(params.ESTIMANDS, checkIfExists: true)
    // density_estimators = Channel.fromPath(params.DENSITY_ESTIMATORS, checkIfExists: true)
    DensityEstimationInputs(dataset, estimands.collect())
    DensityEstimationInputs
        .out
        .splitJson()
        .map{[it.parents, it.outcome]}
        .view{"Item: ${it}"}
}