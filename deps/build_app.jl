using PackageCompiler
PackageCompiler.create_app(".", "popgen",
    executables = ["popgen" => "julia_main"],
    precompile_execution_file="deps/execute.jl", 
    include_lazy_artifacts=true
)
