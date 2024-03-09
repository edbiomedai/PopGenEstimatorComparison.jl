module TestGLM

using Test
using PopGenEstimatorComparison
using Random
using Distributions
using DataFrames
using MLJBase

TESTDIR = joinpath(pkgdir(PopGenEstimatorComparison), "test")

include(joinpath(TESTDIR, "testutils.jl"))

@testset "Test Continuous Outcome GLM" begin
    X, y = make_regression()
    # Constructor based on (X, y)
    estimator = GLMEstimator(X, y)
    train!(estimator, X, y, verbosity=0)
    
end

end

true