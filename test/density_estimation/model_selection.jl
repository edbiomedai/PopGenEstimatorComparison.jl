module TestModelSelection

using Test
using PopGenEstimatorComparison
using PopGenEstimatorComparison: get_treatments
using Random
using Distributions
using DataFrames
using JLD2
using Flux
using JSON
using CSV
using Serialization
using TMLE

TESTDIR = joinpath(pkgdir(PopGenEstimatorComparison), "test")

include(joinpath(TESTDIR, "testutils.jl"))

@testset "Test misc" begin
    density_file = joinpath(TESTDIR, "assets", "conditional_density_x_y.json")
    outcome, features = PopGenEstimatorComparison.read_density_variables(density_file)
    @test outcome == :x
    @test features == [:y]

    # Test make_compatible_estimands_groups
    estimands = [
        ATE(
            outcome=:Y1,
            treatment_values=(T1=(case=1, control=0), ),
            treatment_confounders=(:W1, :W2)
        ),
        ATE(
            outcome=:Y1,
            treatment_values=(T1=(case=1, control=0), T2=(case=1, control=0)),
            treatment_confounders=(:W1, :W2)
        ),
        ATE(
            outcome=:Y1,
            treatment_values=(T2=(case=1, control=0),),
            treatment_confounders=(:W1, :W2),
            outcome_extra_covariates=(:C,)
        ),
        ATE(
            outcome=:Y2,
            treatment_values=(T1=(case=1, control=0),),
            treatment_confounders=(:W1, :W2),
        ),
        ATE(
            outcome=:Y2,
            treatment_values=(T2=(case=1, control=0),),
            treatment_confounders=(:W1,),
        ),
        IATE(
            outcome=:Y1,
            treatment_values=(T1=(case=1, control=0), T2=(case=1, control=0)),
            treatment_confounders=(:W1, :W2)
        ),
        IATE(
            outcome=:Y2,
            treatment_values=(T1=(case=1, control=0), T2=(case=1, control=0)),
            treatment_confounders=(:W1, :W2),
            outcome_extra_covariates=(:C,)
        ),
    ]
    groups = PopGenEstimatorComparison.make_compatible_estimands_groups(estimands)
    # group 1
    @test groups[1].estimands == [estimands[1], estimands[4]]
    @test groups[1].conditional_densities == Dict(
        :Y2 => (:T1, :W1, :W2),
        :Y1 => (:T1, :W1, :W2),
        :T1 => (:W1, :W2)
    )
    # group 2
    @test groups[2].estimands == [estimands[2], estimands[6], estimands[7]]
    @test groups[2].conditional_densities == Dict(
        :Y2 => (:C, :T1, :T2, :W1, :W2),
        :T2 => (:W1, :W2),
        :Y1 => (:T1, :T2, :W1, :W2),
        :T1 => (:W1, :W2)
    )
    # group 3
    @test groups[3].estimands == [estimands[3]]
    @test groups[3].conditional_densities == Dict(
        :T2 => (:W1, :W2),
        :Y1 => (:C, :T2, :W1, :W2)
    )
    # group 4
    @test groups[4].estimands == [estimands[5]]
    @test groups[4].conditional_densities == Dict(
        :Y2 => (:T2, :W1),
        :T2 => (:W1,)
    )
end

@testset "Test density_estimation_inputs" begin
    # With big batchsize
    datasetfile = joinpath(TESTDIR, "assets", "dataset.arrow")
    estimands_prefix = joinpath(TESTDIR, "assets", "estimands", "estimands_")
    outputdir = mktempdir()
    output_prefix = joinpath(outputdir, "de_")
    copy!(ARGS, [
        "density-estimation-inputs",
        datasetfile,
        estimands_prefix,
        string("--output-prefix=", output_prefix),
        string("--batchsize=10")
    ])
    PopGenEstimatorComparison.julia_main()
    # group 1
    estimands = deserialize(joinpath(outputdir, "de_group_1_estimands_1.jls")).estimands
    @test length(estimands) == 3
    cdes = Set([JSON.parsefile(joinpath(outputdir, "de_group_1_conditional_density_$i.json")) for i in 1:4])
    @test cdes == Set([
        Dict("parents" => ["W"], "outcome" => "T₂"),
        Dict("parents" => ["C", "T₁", "W"], "outcome" => "Ycont"),
        Dict("parents" => ["W"], "outcome" => "T₁"),
        Dict("parents" => ["C", "T₁", "T₂", "W"], "outcome" => "Ybin")
    ])
    # group 2
    estimands = deserialize(joinpath(outputdir, "de_group_2_estimands_1.jls")).estimands
    @test only(estimands) == IATE(
        outcome=:Ycont, 
        treatment_values=(T₁ = (case = 1, control = 0), T₂ = (case = 1, control = 0)),
        treatment_confounders=(:W,),
        outcome_extra_covariates=(:C,)
    )
    cdes = Set([JSON.parsefile(joinpath(outputdir, "de_group_2_conditional_density_$i.json")) for i in 1:3])
    @test cdes == Set([
        Dict("parents" => ["W"], "outcome" => "T₂"),
        Dict("parents" => ["W"], "outcome" => "T₁"),
        Dict("parents" => ["C", "T₁", "T₂", "W"], "outcome" => "Ycont")
    ])
    # With batchsize = 1, only ates hence only one group and 2 batches
    estimands_prefix = joinpath(TESTDIR, "assets", "estimands", "estimands_ates")
    outputdir = mktempdir()
    output_prefix = joinpath(outputdir, "de_")
    copy!(ARGS, [
        "density-estimation-inputs",
        datasetfile,
        estimands_prefix,
        string("--output-prefix=", output_prefix),
        string("--batchsize=1")
    ])
    PopGenEstimatorComparison.julia_main()
    @test length(deserialize(joinpath(outputdir, "de_group_1_estimands_1.jls")).estimands) == 1
    @test length(deserialize(joinpath(outputdir, "de_group_1_estimands_2.jls")).estimands) == 1

    cdes = Set([JSON.parsefile(joinpath(outputdir, "de_group_1_conditional_density_$i.json")) for i in 1:4])
    @test cdes == Set([
        Dict("parents" => ["W"], "outcome" => "T₂"),
        Dict("parents" => ["C", "T₁", "W"], "outcome" => "Ycont"),
        Dict("parents" => ["W"], "outcome" => "T₁"),
        Dict("parents" => ["C", "T₁", "T₂", "W"], "outcome" => "Ybin")
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

    density_file = joinpath(TESTDIR, "assets", "conditional_density_x_y.json")
    estimators_file = joinpath(TESTDIR, "assets", "density_estimators.jl")
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
        @test io["best-estimator"] isa NeuralNetworkEstimator
        @test io["outcome"] == :x
        @test io["parents"] == [:y]
        metrics = io["metrics"]
        @test length(metrics) == 3
    end
end

@testset "Test density_estimation: with categorical variables" begin
    outputdir = mktempdir()
    output = joinpath(outputdir, "density_estimate.hdf5")
    datasetfile = joinpath(TESTDIR, "assets", "dataset.arrow")
    density_file = joinpath(TESTDIR, "assets", "conditional_density_Ybin.json")
    estimators_file = joinpath(TESTDIR, "assets", "density_estimators.jl")
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
        @test io["outcome"] == :Ybin
        @test io["parents"] == [:C, :T₁, :W]
        metrics = io["metrics"]
        @test length(metrics) == 3
        @test length(io["estimators"]) == 3
        @test haskey(io, "best-estimator")
    end
end

end

true