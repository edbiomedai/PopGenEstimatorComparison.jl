module TestDensityEstimationSampler

using Test
using PopGenEstimatorComparison

# @testset "Test DensityEstimationSampler" begin
#     n = 100
#     rng = Random.default_rng()
#     Random.seed!(rng, 0)
#     dataset = DataFrame(
#         T₁ = categorical(rand(["AC", "CC", "AA"], n)),
#         T₂ = categorical(rand(["GT", "GG", "TT"], n)),
#         Ybin = categorical(rand([0, 1], n)),
#         Ycont = rand(n),
#         W₁ = rand(n),
#         W₂ = rand(n),
#         C = rand(n)
#     )
#     outcome = "Ycont"
#     treatments = ("T₁", "T₂")
#     sampler = DensityEstimationSampler(dataset, outcome, treatments; 
#         confounders=("W₁", "W₂"), 
#         outcome_extra_covariates=("C", ),
#         hidden_sizes = (20,),
#         K = 3,
#         verbosity=1,
#         batchsize=16,
#         max_epochs=10
#     )
#     sampled_dataset = sample_from(sampler, dataset, n=100)
#     @test names(sampled_dataset) == ["W₁", "W₂", "C", "T₁", "T₂", "Ycont"]
#     @test size(dataset, 1) == 100
# end

end

true
