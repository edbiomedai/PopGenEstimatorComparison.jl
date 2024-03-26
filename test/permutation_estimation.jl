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

@testset "End-to-end Test" begin
    # For some reason if Julia is installed with juliaup on MacOS, the executable is not in ENV["PATH"]
    # Only Test the workflow runs for now
    r = run(addenv(
        `nextflow run main.nf -entry PERMUTATION_ESTIMATION -c test/assets/config/permutation.config -resume`, 
        "PATH" => ENV["PATH"] * ":" * Sys.BINDIR
    ))
    @test r.exitcode == 0

    jldopen(joinpath("results", "permutation_results.hdf5")) do io
        results = io["results"]
        @test size(results) == (16, 4)
        @test Set(results.REPEAT_ID) == Set([1, 2])
        @test Set(results.SAMPLE_SIZE) == Set([100, 200])
        @test Set(results.RNG_SEED) == Set([0])
        # non regression
        @test count(x -> x isa TMLE.Estimate, results.TMLE) > 5
    end
end

end

true