using PackageCompiler
PackageCompiler.create_sysimage(
    ["PopGenEstimatorComparison"], 
    cpu_target="generic",
    sysimage_path="PopGenSysimage.so", 
    precompile_execution_file="deps/execute.jl", 
)
