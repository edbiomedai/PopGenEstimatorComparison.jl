def JuliaCmd() {
    if (workflow.containerEngine == null){
        return "julia --project=${projectDir} --startup-file=no ${projectDir}/popgen.jl"
    }
    else {
        return """
        TEMPD=\$(mktemp -d)
        JULIA_DEPOT_PATH=\$TEMPD:/opt julia --project=/opt/PopGenDRSim --startup-file=no /opt/PopGenDRSim/popgen.jl
        """
    }
}