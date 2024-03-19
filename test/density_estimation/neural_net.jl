module TestNeuralNetEstimation

using Test
using PopGenEstimatorComparison
using PopGenEstimatorComparison: net_train_validation_split, X_y,
    compute_loss, early_stopping_message, train_validation_split
using Random
using Distributions
using DataFrames
using MLJBase
using CategoricalArrays

TESTDIR = joinpath(pkgdir(PopGenEstimatorComparison), "test")

include(joinpath(TESTDIR, "testutils.jl"))

@testset "Test Continuous Outcome MixtureDensityNetwork" begin
    # Seeding
    rng = Random.default_rng()
    Random.seed!(rng, 0)
    dataset = Float32.(sinusoidal_dataset(;n_samples=1000))
    X, y = X_y(dataset, [:x], :y)
    X_train, y_train, X_val, y_val = train_validation_split(X, y)

    # Training and Evaluation
    estimator = NeuralNetworkEstimator(MixtureDensityNetwork(), max_epochs=10_000)
    training_loss_before_train = evaluation_metrics(estimator, X_train, y_train).logloss
    val_loss_before_train = evaluation_metrics(estimator, X_val, y_val).logloss
    @test_logs (:info, early_stopping_message(5)) train!(estimator, X, y, verbosity=1)
    training_loss_after_train = evaluation_metrics(estimator, X_train, y_train).logloss
    val_loss_after_train = evaluation_metrics(estimator, X_val, y_val).logloss
    @test training_loss_before_train > training_loss_after_train
    @test val_loss_before_train > val_loss_after_train
    # Sampling
    y_sampled = sample_from(estimator, X)
    @test y_sampled isa Vector
end

@testset "Test CategoricalMLP" begin
    rng = Random.default_rng()
    Random.seed!(rng, 0)
    n = 1000
    p = 3
    X, y = MLJBase.make_blobs(n, 3)
    X = Float32.(DataFrame(X))
    X_train, y_train, X_val, y_val = train_validation_split(X, y)
    # Training and Evaluation
    estimator = NeuralNetworkEstimator(CategoricalMLP(input_size=3, hidden_sizes=(20, 3)), max_epochs=10_000)
    training_loss_before_train = evaluation_metrics(estimator, X_train, y_train).logloss
    val_loss_before_train = evaluation_metrics(estimator, X_val, y_val).logloss
    @test_logs (:info, early_stopping_message(5)) train!(estimator, X, y, verbosity=1)
    training_loss_after_train = evaluation_metrics(estimator, X_train, y_train).logloss
    val_loss_after_train = evaluation_metrics(estimator, X_val, y_val).logloss
    @test training_loss_before_train > training_loss_after_train
    @test val_loss_before_train > val_loss_after_train
    # Sample
    y_sampled = sample_from(estimator, X, levels(y))
    @test y_sampled isa CategoricalArrays.CategoricalVector
    @test levels(y_sampled) == levels(y)
end

end
true