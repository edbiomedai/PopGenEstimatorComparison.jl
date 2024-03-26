using PopGenEstimatorComparison
using Test

TESTDIR = joinpath(pkgdir(PopGenEstimatorComparison), "test")

@testset "PopGenEstimatorComparison.jl" begin
    @test include(joinpath(TESTDIR, "utils.jl"))
    
    @test include(joinpath(TESTDIR, "density_estimation", "glm.jl"))
    @test include(joinpath(TESTDIR, "density_estimation", "neural_net.jl"))
    @test include(joinpath(TESTDIR, "density_estimation", "model_selection.jl"))

    @test include(joinpath(TESTDIR, "samplers", "permutation_null_sampler.jl"))
    @test include(joinpath(TESTDIR, "samplers", "density_estimate_sampler.jl"))

    @test include(joinpath(TESTDIR, "permutation_estimation.jl"))
    @test include(joinpath(TESTDIR, "density_estimation.jl"))
end

true
