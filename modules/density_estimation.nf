include { JuliaCmd } from '../modules/functions.nf'

process DensityEstimation {
    label 'multithreaded'
    label 'bigmem'
    publishDir "${params.OUTDIR}/density_estimation/density_estimates", mode: 'symlink'

    input:
        path dataset 
        path density
        
    output:
        path outfile

    script:
        file_splits = density.name.split("_")
        prefix = file_splits[0..-2].join("_")
        file_id = file_splits[-1][0..-6]
        outfile = "${prefix}_estimate_${file_id}.hdf5"
        """
        ${JuliaCmd()} density-estimation ${dataset} ${density} \
            --mode=study \
            --output=${outfile} \
            --train-ratio=${params.TRAIN_RATIO} \
            --verbosity=${params.VERBOSITY}
        """
}