include { JuliaCmd } from './functions.nf'

process AggregateResults {
    label "bigmem"
    publishDir "${params.OUTDIR}", mode: 'symlink'

    input:
        path results
        val outfile
        
    output:
        path outfile

    script:
        """
        ${JuliaCmd()} aggregate results ${outfile}
        """
}