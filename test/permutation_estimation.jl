module TestEstimation

###########################################################
###        PERMUTATION_ESTIMATION WORKFLOW TESTS        ###
###########################################################
# Details
# -------
# This module tests the PERMUTATION_ESTIMATION Workflow
# This workflow is itself composed of two steps:
#   - permutation estimation
#   - aggregation of results
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

PKGDIR = pkgdir(PopGenEstimatorComparison)

TESTDIR = joinpath(PKGDIR, "test")

include(joinpath(TESTDIR, "testutils.jl"))

@testset "Integration Test Permutation Estimation" begin
    outdir = mktempdir()
    nrepeats = 2
    dataset_file = joinpath(TESTDIR, "assets", "dataset.arrow")
    # Run 1: vanilla-glmnet / ATEs / sample-size=100
    workdir1 = mktempdir()
    out1 = joinpath(outdir, "permutation_results_1.hdf5")
    copy!(ARGS, [
        "estimation",
        dataset_file,
        joinpath(TESTDIR, "assets", "estimands", "estimands_ates.jls"),
        joinpath(PKGDIR, "assets", "estimators-configs", "vanilla-glmnet.jl"),
        "--sample-size=100",
        string("--n-repeats=", nrepeats),
        "--rng=0",
        string("--out=", out1),
        string("--workdir=", workdir1)
    ])
    PopGenEstimatorComparison.julia_main()
    jldopen(out1) do io
        @test io["sample_size"] == 100
        @test io["estimators"] == (:wTMLE_GLMNET, :TMLE_GLMNET, :OSE_GLMNET)
        @test length(io["statistics_by_repeat_id"]) == 2
        @test names(io["results"]) == ["wTMLE_GLMNET", "TMLE_GLMNET", "OSE_GLMNET", "REPEAT_ID", "RNG_SEED"]
        @test size(io["results"]) == (6, 5)
    end
    # Run 2: vanilla-xgboost / ATEs / sample-size=200
    workdir2 = mktempdir()
    out2 = joinpath(outdir, "permutation_results_2.hdf5")
    copy!(ARGS, [
        "estimation",
        dataset_file,
        joinpath(TESTDIR, "assets", "estimands", "estimands_ates.jls"),
        joinpath(PKGDIR, "assets", "estimators-configs", "vanilla-xgboost.jl"),
        "--sample-size=200",
        string("--n-repeats=", nrepeats),
        "--rng=1",
        string("--out=", out2),
        string("--workdir=", workdir2)
    ])
    PopGenEstimatorComparison.julia_main()
    jldopen(out2) do io
        @test io["sample_size"] == 200
        @test io["estimators"] == (:wTMLE_XGBOOST, :TMLE_XGBOOST, :OSE_XGBOOST)
        @test length(io["statistics_by_repeat_id"]) == 2
        @test names(io["results"]) == ["wTMLE_XGBOOST", "TMLE_XGBOOST", "OSE_XGBOOST", "REPEAT_ID", "RNG_SEED"]
        @test size(io["results"]) == (6, 5)
    end
    # Run 3: vanilla-xgboost / ATEs / sample-size=200 / 4 repeats
    workdir3 = mktempdir()
    out3 = joinpath(outdir, "permutation_results_3.hdf5")
    copy!(ARGS, [
        "estimation",
        dataset_file,
        joinpath(TESTDIR, "assets", "estimands", "estimands_ates.jls"),
        joinpath(PKGDIR, "assets", "estimators-configs", "vanilla-xgboost.jl"),
        "--sample-size=200",
        string("--n-repeats=", nrepeats),
        "--rng=2",
        string("--out=", out3),
        string("--workdir=", workdir3)
    ])
    PopGenEstimatorComparison.julia_main()
    jldopen(out3) do io
        @test io["sample_size"] == 200
        @test io["estimators"] == (:wTMLE_XGBOOST, :TMLE_XGBOOST, :OSE_XGBOOST)
        @test length(io["statistics_by_repeat_id"]) == 2
        @test names(io["results"]) == ["wTMLE_XGBOOST", "TMLE_XGBOOST", "OSE_XGBOOST", "REPEAT_ID", "RNG_SEED"]
        @test size(io["results"]) == (6, 5)
    end
    # Aggregate the 3 runs
    results_file = joinpath(outdir, "permutation_results.hdf5")
    copy!(ARGS, [
        "aggregate",
        joinpath(outdir, "permutation_results"),
        results_file,
    ])
    PopGenEstimatorComparison.julia_main()
    jldopen(results_file) do io
        results = io["results"]
        run_1 = results[(:wTMLE_GLMNET, :TMLE_GLMNET, :OSE_GLMNET)][100]
        @test names(run_1) == ["wTMLE_GLMNET", "TMLE_GLMNET", "OSE_GLMNET", "REPEAT_ID", "RNG_SEED"]
        @test size(run_1) == (6, 5)
        @test run_1.REPEAT_ID == [1, 1, 1, 2, 2, 2]
        @test run_1.RNG_SEED ==[0, 0, 0, 0, 0, 0]
        run_2_3 = results[(:wTMLE_XGBOOST, :TMLE_XGBOOST, :OSE_XGBOOST)][200]
        @test names(run_2_3) == ["wTMLE_XGBOOST", "TMLE_XGBOOST", "OSE_XGBOOST", "REPEAT_ID", "RNG_SEED"]
        @test size(run_2_3) == (12, 5)
        @test run_2_3.REPEAT_ID == [1, 1, 1, 2, 2, 2, 1, 1, 1, 2, 2, 2]
        @test run_2_3.RNG_SEED ==[1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2]
    end
    # Analyse results
    out_dir = mktempdir()
    copy!(ARGS, [
        "analyse",
        results_file,
        joinpath(TESTDIR, "assets", "estimands", "estimands_ates.jls"),
        string("--out-dir=", out_dir)
    ])
    PopGenEstimatorComparison.julia_main()
    analysis_results = jldopen(io -> io["results"], joinpath(out_dir, "analysis1D", "summary_stats.hdf5"))
    @test names(analysis_results) == ["ESTIMAND", "ESTIMATOR", "SAMPLE_SIZE", "BIAS", "VARIANCE", "MSE", "COVERAGE", "CI_WIDTH"]
end

end

true