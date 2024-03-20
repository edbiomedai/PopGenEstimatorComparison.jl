def JuliaCmd() {
    if (workflow.containerEngine == null){
        return "julia --project=${projectDir} --startup-file=no ${projectDir}/popgen.jl"
    }
    else {
        return """
        TEMPD=\$(mktemp -d)
        JULIA_DEPOT_PATH=\$TEMPD:/opt julia --project=/opt/PopGenDRSim --startup-file=no --sysimage=/opt/PopGenDRSim/PopGenSysimage.so /opt/PopGenDRSim/popgen.jl"""
    }
}

def LongestPrefix(files){
    // Only one file, strangely it is not passed as a list
    if (files instanceof Collection == false) {
        return files.getName()
    }
    // More than one file
    def index = 0
    while(true){
        def current_prefix = files[0].getName()[0..index]
        for (file in files){
            if(file.getName()[0..index] != current_prefix){
                return current_prefix[0..-2]
            }
        }
        index++
    }
}