def JuliaCmd() {
    if (workflow.containerEngine == null){
        return "julia --project=${projectDir} --startup-file=no ${projectDir}/popgen.jl"
    }
    else {
        return "Oopsies..."
    }
}