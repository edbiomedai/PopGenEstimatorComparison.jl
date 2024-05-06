module TestDensityEstimation

###########################################################
###        DENSITY_ESTIMATION WORKFLOW TESTS        ###
###########################################################
# Details
# -------
# This module tests the DENSITY_ESTIMATION Workflow
# This workflow is itself composed of four steps:
#   - density estimation inputs generation: conditional densities to be estimated and target estimands
#   - density estimation
#   - estimation from density estimates
#   - aggregate results
#
# We first test each script command independently, then make sure the Nextflow workflow runs

using Test
using PopGenEstimatorComparison
using TMLE
using Random
using JLD2
using Distributions
using LogExpFunctions
using DataFrames
using JLD2
using JSON
using MLJBase

PKGDIR = pkgdir(PopGenEstimatorComparison)

TESTDIR = joinpath(PKGDIR, "test")

include(joinpath(TESTDIR, "testutils.jl"))

@testset "Integration Test Density Estimation" begin
    dataset_file = joinpath(TESTDIR, "assets", "dataset.arrow")
    estimands_prefix = joinpath(TESTDIR, "assets", "estimands", "estimands_ate")
    # Inputs generation
    de_inputs_dir = mktempdir()
    copy!(ARGS, [
        "density-estimation-inputs",
        dataset_file,
        estimands_prefix,
        "--batchsize=2",
        string("--output-prefix=", joinpath(de_inputs_dir, "de_"))
    ]
    )
    PopGenEstimatorComparison.julia_main()
    # Density Estimation
    density_estimators_file = joinpath(TESTDIR, "assets", "density_estimators.jl")
    density_estimates_dir = mktempdir()
    for density_file in readdir(de_inputs_dir)
        if endswith(density_file, "json")
            file_splits = split(density_file, "_")
            file_id = file_splits[end][1:end-4]
            outfile = joinpath(
                density_estimates_dir,
                string(join(file_splits[1:end-1], "_"), "_estimate_", file_id, "hdf5")
                )
            copy!(ARGS, [
                "density-estimation",
                dataset_file,
                joinpath(de_inputs_dir, density_file),
                "--mode=test",
                string("--output=", outfile),
                "--train-ratio=7",
                "--verbosity=0"
            ])
            PopGenEstimatorComparison.julia_main()
            best_estimator = PopGenEstimatorComparison.sieve_neural_net_density_estimator(outfile)
            @test !(best_estimator isa Nothing)
            density_dict = JSON.parsefile(joinpath(de_inputs_dir, density_file))
            jldopen(outfile) do io
                @test io["outcome"] == Symbol(density_dict["outcome"])
                @test io["parents"] == Symbol.(density_dict["parents"])
                @test all(haskey(v, :train_loss) for v in io["metrics"])
                @test all(haskey(v, :test_loss) for v in io["metrics"])
            end
        end
    end
    # Estimation From Densities 1: glmnet, rng=1
    estimators_config = joinpath(PKGDIR, "assets", "estimators-configs", "vanilla-glmnet.jl")
    outdir = mktempdir()
    estimands_file = joinpath(de_inputs_dir, "de_group_1_estimands_1.jls")
    workdir_1 = mktempdir()
    results_file_1 = joinpath(outdir, "results_1.hdf5")
    copy!(ARGS, [
        "estimation",
        dataset_file,
        estimands_file,
        estimators_config,
        "--sample-size=100",
        string("--density-estimates-prefix=", joinpath(density_estimates_dir, "de")),
        "--n-repeats=2",
        string("--out=", results_file_1),
        "--verbosity=0",
        "--rng=1",
        "--chunksize=100",
        string("--workdir=", workdir_1)
    ])
    PopGenEstimatorComparison.julia_main()
    jldopen(results_file_1) do io
        @test io["estimators"] == (:wTMLE_GLMNET, :TMLE_GLMNET, :OSE_GLMNET)
        # 1 repeat
        @test length(io["statistics_by_repeat_id"]) == 2
        @test io["sample_size"] == 100
        results = io["results"]
        @test results.REPEAT_ID == [1, 1, 2, 2]
        @test results.RNG_SEED == [1, 1, 1, 1]
        # At least 2 successes
        @test count(x -> x isa TMLE.Estimate, results.TMLE_GLMNET) > 2
    end
    # Estimation From Densities 2: glmnet, rng=2
    workdir_2 = mktempdir()
    results_file_2 = joinpath(outdir, "results_2.hdf5")
    copy!(ARGS, [
        "estimation",
        dataset_file,
        estimands_file,
        estimators_config,
        "--sample-size=100",
        string("--density-estimates-prefix=", joinpath(density_estimates_dir, "de")),
        "--n-repeats=2",
        string("--out=", results_file_2),
        "--verbosity=0",
        "--rng=2",
        "--chunksize=100",
        string("--workdir=", workdir_2)
    ])
    PopGenEstimatorComparison.julia_main()
    jldopen(results_file_2) do io
        @test io["estimators"] == (:wTMLE_GLMNET, :TMLE_GLMNET, :OSE_GLMNET)
        # 1 repeat
        @test length(io["statistics_by_repeat_id"]) == 2
        @test io["sample_size"] == 100
        results = io["results"]
        @test results.REPEAT_ID == [1, 1, 2, 2]
        @test results.RNG_SEED == [2, 2, 2, 2]
        # At least 2 successes
        @test count(x -> x isa TMLE.Estimate, results.TMLE_GLMNET) > 2
    end
    # Aggregate
    results_file = joinpath(outdir, "from_densities_results.hdf5")
    copy!(ARGS, [
        "aggregate",
        joinpath(outdir, "results"),
        results_file,
    ])
    PopGenEstimatorComparison.julia_main()
    jldopen(results_file) do io
        results = io["results"]
        run_100 = results[(:wTMLE_GLMNET, :TMLE_GLMNET, :OSE_GLMNET)][100]
        @test names(run_100) == ["wTMLE_GLMNET", "TMLE_GLMNET", "OSE_GLMNET", "REPEAT_ID", "RNG_SEED"]
        @test size(run_100) == (8, 5)
        @test run_100.REPEAT_ID == [1, 1, 2, 2, 1, 1, 2, 2]
        @test run_100.RNG_SEED ==[1, 1, 1, 1, 2, 2, 2, 2]
    end
    # Analysis
    analysis_outdir = mktempdir()
    copy!(ARGS, [
        "analyse",
        results_file,
        estimands_prefix,
        string("--out-dir=", analysis_outdir),
        string("--n=", 100_000),
        string("--dataset-file=", dataset_file),
        string("--density-estimates-prefix=", joinpath(density_estimates_dir, "de_group")),

    ])
    PopGenEstimatorComparison.julia_main()
    analysis_results = jldopen(io -> io["results"], joinpath(analysis_outdir, "analysis1D", "summary_stats.hdf5"))
    @test names(analysis_results) == ["ESTIMAND", "ESTIMATOR", "SAMPLE_SIZE", "BIAS", "VARIANCE", "MSE", "COVERAGE", "CI_WIDTH"]
    @test isfile(joinpath(analysis_outdir, "om_density_estimation.png"))
end


end

true