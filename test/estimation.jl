module TestEstimation

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

@testset "Test estimate_from_simulated_dataset" begin
    outdir = mktempdir()
    # Run 1
    out1 = joinpath(outdir, "permutation_results_1.hdf5")
    copy!(ARGS, [
        "permutation-estimation",
        joinpath(TESTDIR, "assets", "dataset.arrow"),
        joinpath(TESTDIR, "assets", "estimands", "ates.jls"),
        joinpath(PKGDIR, "assets", "estimators-configs", "glm.jl"),
        "--sample-size=100",
        "--n-repeats=2",
        "--rng=0",
        string("--out=", out1)
    ])
    PopGenEstimatorComparison.julia_main()
    # Run 2
    out2 = joinpath(outdir, "permutation_results_2.hdf5")
    copy!(ARGS, [
        "permutation-estimation",
        joinpath(TESTDIR, "assets", "dataset.arrow"),
        joinpath(TESTDIR, "assets", "estimands", "ates.jls"),
        joinpath(PKGDIR, "assets", "estimators-configs", "glm.jl"),
        "--sample-size=200",
        "--n-repeats=2",
        "--rng=1",
        string("--out=", out2)
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

    results = jldopen(io -> io["results"], out)
    @test names(results) == ["TMLE", "REPEAT_ID", "SAMPLE_SIZE", "RNG_SEED"]
    @test size(results, 1) == 2*2*2
    @test results.RNG_SEED == [0, 0, 0, 0, 1, 1, 1, 1]
    @test results.REPEAT_ID == [1, 1, 2, 2, 1, 1, 2, 2]
    @test results.SAMPLE_SIZE == [100, 100, 100, 100, 200, 200, 200, 200]
end

end

true