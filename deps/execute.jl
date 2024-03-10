using PopGenEstimatorComparison

@info "Running precompilation script."
# Run workload
TEST_DIR = joinpath(pkgdir(PopGenEstimatorComparison), "test")
push!(LOAD_PATH, TEST_DIR)
include(joinpath(TEST_DIR, "runtests.jl"))