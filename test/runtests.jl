using PopGenEstimatorComparison
using Test

TESTDIR = joinpath(pkgdir(PopGenEstimatorComparison), "test")

@testset "PopGenEstimatorComparison.jl" begin
    @test include(joinpath(TESTDIR, "generative_models.jl"))
    @test include(joinpath(TESTDIR, "estimation.jl"))
end
