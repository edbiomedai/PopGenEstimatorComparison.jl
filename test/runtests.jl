using PopGenEstimatorComparison
using Test

TESTDIR = joinpath(pkgdir(PopGenEstimatorComparison), "test")

@testset "PopGenEstimatorComparison.jl" begin
    @test include(joinpath(TESTDIR, "utils.jl"))
    include(joinpath(TESTDIR, "density_estimation", "glm.jl"))
    include(joinpath(TESTDIR, "density_estimation", "neural_net.jl"))
    include(joinpath(TESTDIR, "density_estimation", "model_selection.jl"))

    include(joinpath(TESTDIR, "samplers", "permutation_null_sampler.jl"))
    include(joinpath(TESTDIR, "samplers", "density_estimation_sampler.jl"))

    # @test include(joinpath(TESTDIR, "estimation.jl"))
end
