module TestEstimation

using Test
using PopGenEstimatorComparison
using TMLE
using Random
using JLD2
using Distributions
using LogExpFunctions
using DataFrames
using TOML

TESTDIR = pkgdir(PopGenEstimatorComparison, "test")

include(joinpath(TESTDIR, "testutils.jl"))

@testset "Test estimate_from_simulated_dataset" begin
    origin_dataset = linear_interaction_dataset(100)
    estimands_config = linear_interaction_dataset_ATEs()
    sample_size = 100
    rng_seed = 0
    nrepeats = 2
    estimators = (TMLE=TMLEE(), OSE=OSE())
    outdir1 = mktempdir()
    permutation_sampling_estimation(origin_dataset, estimands_config, estimators, sample_size;
        nrepeats=nrepeats,
        outdir=outdir1,    
        verbosity=0,
        rng_seed=rng_seed
    )
    outdir2 = mktempdir()
    sample_size = 200
    rng_seed = 1
    permutation_sampling_estimation(origin_dataset, estimands_config, estimators, sample_size;
        nrepeats=nrepeats,
        outdir=outdir2,    
        verbosity=0,
        rng_seed = rng_seed
    )

    results = read_results_dirs(outdir1, outdir2)
    @test names(results) == ["TMLE", "OSE", "REPEAT_ID", "SAMPLE_SIZE", "RNG_SEED"]
    @test size(results, 1) == 2*2*2
    @test results.RNG_SEED == [0, 0, 0, 0, 1, 1, 1, 1]
    @test results.REPEAT_ID == [1, 1, 2, 2, 1, 1, 2, 2]
    @test results.SAMPLE_SIZE == [100, 100, 100, 100, 200, 200, 200, 200]
end

end

true