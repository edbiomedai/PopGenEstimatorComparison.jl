include { JuliaCmd } from './functions.nf'

process AggregateResults {
    publishDir "${params.OUTDIR}", mode: 'symlink'

    input:
        path results
        val outfile
        
    output:
        path outfile

    script:
        """
        ${JuliaCmd()} aggregate permutation_results ${outfile}
        """
}