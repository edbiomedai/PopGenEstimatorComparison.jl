module TestAnalysis

using Test
using PopGenEstimatorComparison

TESTDIR = joinpath(pkgdir(PopGenEstimatorComparison), "test")

@testset "Test true effects" begin
    # Null Sampler
    estimands_prefix = joinpath(TESTDIR, "assets", "estimands", "estimands")
    true_effects = PopGenEstimatorComparison.get_true_effect_sizes(estimands_prefix, nothing, nothing; n=10)
    @test true_effects.TRUE_EFFECT == [0, [0, 0], 0, 0]
    # DensityEstimateSampler
end

@testset "Test analysis" begin
    estimation_results_file = joinpath(TESTDIR, "assets", "permutation_results.hdf5")

    dataset_file = joinpath(TESTDIR, "assets", "dataset.arrow")
    copy!(ARGS, [

    ])
    PopGenEstimatorComparison.julia_main()
end

end

true