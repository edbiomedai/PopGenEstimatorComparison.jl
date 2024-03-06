module TestModelSelection

using Test
using PopGenEstimatorComparison
using Random
using Distributions
using DataFrames

TESTDIR = joinpath(pkgdir(PopGenEstimatorComparison), "test")

include(joinpath(TESTDIR, "testutils.jl"))

@testset "Test evaluate_and_save_density_estimators" begin
    outpath, _ = mktemp()
    rng = Random.default_rng()
    Random.seed!(rng, 0)
    dataset = Float32.(sinusoidal_dataset(;n_samples=1000))
    density_estimators, metrics = evaluate_and_save_density_estimators!(
        dataset, 
        :x, 
        [:y];
        outpath=outpath,
        train_ratio=10,
        verbosity=1
    )

end

end

true