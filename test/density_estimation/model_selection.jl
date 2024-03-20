module TestModelSelection

using Test
using PopGenEstimatorComparison
using Random
using Distributions
using DataFrames
using JLD2
using Flux
using JSON
using CSV

TESTDIR = joinpath(pkgdir(PopGenEstimatorComparison), "test")

include(joinpath(TESTDIR, "testutils.jl"))

@testset "Test misc" begin
    density_file = joinpath("test", "assets", "conditional_density_x_y.json")
    outcome, features = PopGenEstimatorComparison.read_density_variables(density_file)
    @test outcome == "x"
    @test features == ["y"]
end
@testset "Test density_estimation_inputs" begin
    datasetfile = "test/assets/dataset.arrow"
    estimands_prefix = joinpath("test", "assets", "estimands", "estimands_")
    outputdir = mktempdir()
    output_prefix = joinpath(outputdir, "conditional_density_")
    copy!(ARGS, [
        "density-estimation-inputs",
        datasetfile,
        estimands_prefix,
        string("--output-prefix=", output_prefix)
    ])
    PopGenEstimatorComparison.julia_main()
    conditional_densities_variables = reduce(vcat, JSON.parsefile(f) for f in readdir(outputdir, join=true))
    @test Set(conditional_densities_variables) == Set([
        Dict{String, Any}("parents" => Any["C", "T₁", "W"], "outcome" => "Ycont"),
        Dict{String, Any}("parents" => Any["W"], "outcome" => "T₂"),
        Dict{String, Any}("parents" => Any["C", "T₁", "T₂", "W"], "outcome" => "Ybin"),
        Dict{String, Any}("parents" => Any["C", "T₁", "T₂", "W"], "outcome" => "Ycont"),
        Dict{String, Any}("parents" => Any["W"], "outcome" => "T₁")
    ])
end

@testset "Test density_estimation: sinusoidal problem" begin
    # On this dataset, the GLM has no chance to perform best
    rng = Random.default_rng()
    Random.seed!(rng, 0)
    dataset = sinusoidal_dataset(;n_samples=1000)

    outputdir = mktempdir()
    output = joinpath(outputdir, "density_estimate.hdf5")
    datasetfile = joinpath(outputdir, "dataset.csv")

    CSV.write(datasetfile, dataset)

    density_file = joinpath("test", "assets", "conditional_density_x_y.json")
    estimators_file = joinpath("test", "assets", "density_estimators.jl")
    copy!(ARGS, [
        "density-estimation",
        datasetfile,
        density_file,
        string("--estimators=", estimators_file),
        string("--output=", output),
        string("--train-ratio=10"),
        string("--verbosity=0")
    ])
    PopGenEstimatorComparison.julia_main()

    jldopen(output) do io
        @test io["outcome"] == "x"
        @test io["features"] == ["y"]
        metrics = io["metrics"]
        @test length(metrics) == 3
        best_de = PopGenEstimatorComparison.best_density_estimator(io["estimators"], metrics)
        @test best_de isa PopGenEstimatorComparison.NeuralNetworkEstimator
    end
end

@testset "Test density_estimation: with categorical variables" begin
    outputdir = mktempdir()
    output = joinpath(outputdir, "density_estimate.hdf5")
    datasetfile = joinpath("test", "assets", "dataset.arrow")
    density_file = joinpath("test", "assets", "conditional_density_Ybin.json")
    estimators_file = joinpath("test", "assets", "density_estimators.jl")
    copy!(ARGS, [
        "density-estimation",
        datasetfile,
        density_file,
        string("--estimators=", estimators_file),
        string("--output=", output),
        string("--train-ratio=10"),
        string("--verbosity=0")
    ])
    PopGenEstimatorComparison.julia_main()

    jldopen(output) do io
        @test io["outcome"] == "Ybin"
        @test io["features"] == ["T₁", "C", "W"]
        metrics = io["metrics"]
        @test length(metrics) == 3
        @test length(io["estimators"]) == 3
    end
end

end

true