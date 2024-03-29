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

function test_results(out)
    results = jldopen(io -> io["results"], out)
    @test names(results) == ["TMLE", "REPEAT_ID", "SAMPLE_SIZE", "RNG_SEED"]
    @test size(results, 1) == 2*2*2
    @test results.RNG_SEED == [0, 0, 0, 0, 1, 1, 1, 1]
    @test results.REPEAT_ID == [1, 1, 2, 2, 1, 1, 2, 2]
    @test results.SAMPLE_SIZE == [100, 100, 100, 100, 200, 200, 200, 200]
end

@testset "Integration Test" begin
    outdir = mktempdir()
    dataset_file = joinpath(TESTDIR, "assets", "dataset.arrow")
    # Run 1
    workdir1 = mktempdir()
    out1 = joinpath(outdir, "permutation_results_1.hdf5")
    copy!(ARGS, [
        "estimation",
        dataset_file,
        joinpath(TESTDIR, "assets", "estimands", "estimands_ates.jls"),
        joinpath(PKGDIR, "assets", "estimators-configs", "glm.jl"),
        "--sample-size=100",
        "--n-repeats=2",
        "--rng=0",
        string("--out=", out1),
        string("--workdir=", workdir1)
    ])
    PopGenEstimatorComparison.julia_main()
    # Run 2
    workdir2 = mktempdir()
    out2 = joinpath(outdir, "permutation_results_2.hdf5")
    copy!(ARGS, [
        "estimation",
        joinpath(TESTDIR, "assets", "dataset.arrow"),
        joinpath(TESTDIR, "assets", "estimands", "estimands_ates.jls"),
        joinpath(PKGDIR, "assets", "estimators-configs", "glm.jl"),
        "--sample-size=200",
        "--n-repeats=2",
        "--rng=1",
        string("--out=", out2),
        string("--workdir=", workdir2)
    ])
    PopGenEstimatorComparison.julia_main()
    # Aggregate the 2 runs
    out = joinpath(outdir, "permutation_results.hdf5")
    copy!(ARGS, [
        "aggregate",
        joinpath(outdir, "permutation_results"),
        out,
    ])
    PopGenEstimatorComparison.julia_main()
    test_results(out)
end

end

true