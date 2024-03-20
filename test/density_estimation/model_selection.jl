module TestModelSelection

using Test
using PopGenEstimatorComparison
using Random
using Distributions
using DataFrames
using JLD2
using Flux
using JSON

TESTDIR = joinpath(pkgdir(PopGenEstimatorComparison), "test")

include(joinpath(TESTDIR, "testutils.jl"))

@testset "Test density_estimation_inputs" begin
    datasetfile = "test/assets/dataset.arrow"
    estimands_prefix = joinpath("test", "assets", "estimands", "estimands_")
    output, _ = mktemp()
    density_estimation_inputs(datasetfile, estimands_prefix; output=output)

    conditional_densities_variables = JSON.parsefile(output)
    @test conditional_densities_variables == [
        Dict{String, Any}("parents" => Any["C", "T₁", "W"], "outcome" => "Ycont"),
        Dict{String, Any}("parents" => Any["W"], "outcome" => "T₂"),
        Dict{String, Any}("parents" => Any["C", "T₁", "T₂", "W"], "outcome" => "Ybin"),
        Dict{String, Any}("parents" => Any["C", "T₁", "T₂", "W"], "outcome" => "Ycont"),
        Dict{String, Any}("parents" => Any["W"], "outcome" => "T₁")
    ]
end

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