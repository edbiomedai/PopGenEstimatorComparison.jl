module TestGLM

using Test
using PopGenEstimatorComparison
using Random
using Distributions
using DataFrames
using MLJBase
using MLJGLMInterface
using MLJLinearModels
using CategoricalArrays

TESTDIR = joinpath(pkgdir(PopGenEstimatorComparison), "test")

include(joinpath(TESTDIR, "testutils.jl"))

@testset "Test Continuous Outcome GLM" begin
    X, y = make_regression()
    # Constructor based on (X, y)
    estimator = GLMEstimator(X, y)
    @test estimator.model isa ProbabilisticPipeline
    @test estimator.model.linear_regressor isa MLJGLMInterface.LinearRegressor
    # Training
    train!(estimator, X, y, verbosity=0)
    @test fitted_params(estimator.machine) isa NamedTuple
    # Sampling
    y_sampled = sample_from(estimator, X)
    @test y_sampled isa Vector
    # Evaluate 
    metrics = evaluation_metrics(estimator, X, y)
    @test metrics isa NamedTuple{(:logloss,)}
end

@testset "Test Categorical Outcome GLM" begin
    X, y = make_blobs()
    # Constructor based on (X, y)
    estimator = GLMEstimator(X, y)
    @test estimator.model isa ProbabilisticPipeline
    @test estimator.model.logistic_classifier isa MLJLinearModels.LogisticClassifier
    # Training
    train!(estimator, X, y, verbosity=0)
    @test fitted_params(estimator.machine) isa NamedTuple
    # Sampling
    y_sampled = sample_from(estimator, X)
    @test y_sampled isa CategoricalVector
    # Evaluate 
    metrics = evaluation_metrics(estimator, X, y)
    @test metrics isa NamedTuple{(:logloss,)}
end

end

true