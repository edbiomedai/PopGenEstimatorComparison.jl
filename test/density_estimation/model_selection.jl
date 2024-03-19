module TestModelSelection

using Test
using PopGenEstimatorComparison
using Random
using Distributions
using DataFrames
using JLD2
using Flux

TESTDIR = joinpath(pkgdir(PopGenEstimatorComparison), "test")

include(joinpath(TESTDIR, "testutils.jl"))

@testset "Test density_estimation" begin
    # On this dataset, the GLM has no chance to perform best
    output, _ = mktemp()
    rng = Random.default_rng()
    Random.seed!(rng, 0)
    dataset = sinusoidal_dataset(;n_samples=1000)
    density_estimation(
        dataset, 
        :x, 
        [:y];
        output=output,
        train_ratio=10,
        verbosity=1
    )
    jldopen(output) do io
        metrics = io["metrics"]
        @test length(metrics) == 3
        best_de = PopGenEstimatorComparison.best_density_estimator(io["estimators"], metrics)
        @test best_de isa PopGenEstimatorComparison.NeuralNetworkEstimator
    end

end

end

true