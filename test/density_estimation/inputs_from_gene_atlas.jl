module TestInputsFromGeneAtlas

using TMLE
using Test
using PopGenEstimatorComparison

TESTDIR = joinpath(pkgdir(PopGenEstimatorComparison), "test")

include(joinpath(TESTDIR, "testutils.jl"))

@testset "Test misc" begin
    estimands = linear_interaction_dataset_ATEs().estimands
    trait_to_variants = PopGenEstimatorComparison.get_trait_to_variants_from_estimands(estimands; regex=r"")
    
end

end

true