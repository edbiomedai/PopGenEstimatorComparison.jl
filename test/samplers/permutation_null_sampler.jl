module TestPermutationNullSampler

using Test
using PopGenEstimatorComparison
using Random
using DataFrames
using CategoricalArrays
using Statistics

TESTDIR = pkgdir(PopGenEstimatorComparison, "test")

include(joinpath(TESTDIR, "testutils.jl"))

@testset "Test PermutationNullSampler" begin
    rng = Random.default_rng()
    Random.seed!(rng, 0)
    origin_dataset = dummy_dataset()
    sampler = PermutationNullSampler(:Ycont, [:T₁, :T₂]; 
        confounders=[:W₁, :W₂],
        outcome_extra_covariates=[:C]
    )
    variables = sampler.variables
    @test variables.outcome == :Ycont
    @test variables.treatments == (:T₁, :T₂)
    @test variables.confounders == (:W₁, :W₂)
    @test variables.outcome_extra_covariates == (:C, )
    sampled_dataset = sample_from(sampler, origin_dataset, n=1000)
    @test names(sampled_dataset) == ["W₁", "W₂", "C", "T₁", "T₂", "Ycont"]
    @test size(sampled_dataset, 1) == 1000
    # Structure between (W, C) is preserved
    origin_WC = [row for row in eachrow(origin_dataset[!, ["W₁", "W₂", "C"]])]
    @test all(row in origin_WC for row in eachrow(sampled_dataset[!, ["W₁", "W₂", "C"]]))
    # Basic stats are somewhat preserved
    @test mean(sampled_dataset.Ycont) ≈ mean(origin_dataset.Ycont) atol=0.1
    for T in (:T₁, :T₂)
        p_sampled_T = sort(combine(groupby(sampled_dataset, T), proprow), :proprow)
        p_origin_T = sort(combine(groupby(origin_dataset, T), proprow), :proprow)
        @test p_sampled_T[!, T] == p_origin_T[!, T]
    end
end

end

true