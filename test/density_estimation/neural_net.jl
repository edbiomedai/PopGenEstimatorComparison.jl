module TestNeuralNetEstimation

using Test
using PopGenEstimatorComparison
using PopGenEstimatorComparison: X_y, early_stopping_message, 
    train_validation_split
using Random
using Distributions
using DataFrames
using MLJBase
using CategoricalArrays
using TMLE

TESTDIR = joinpath(pkgdir(PopGenEstimatorComparison), "test")

include(joinpath(TESTDIR, "testutils.jl"))

@testset "Test Continuous Outcome MixtureDensityNetwork" begin
    # Seeding
    rng = Random.default_rng()
    Random.seed!(rng, 0)
    n_samples = 1000
    dataset = Float32.(sinusoidal_dataset(;n_samples=n_samples))
    X, y = X_y(dataset, [:x], :y)
    X_train, y_train, X_val, y_val = train_validation_split(X, y)

    # Training and Evaluation
    estimator = NeuralNetworkEstimator(X, y, max_epochs=10_000)
    training_loss_before_train = evaluation_metrics(estimator, X_train, y_train).logloss
    val_loss_before_train = evaluation_metrics(estimator, X_val, y_val).logloss
    estimator, info = train!(estimator, X, y, verbosity=0)
    training_loss_after_train = evaluation_metrics(estimator, X_train, y_train).logloss
    val_loss_after_train = evaluation_metrics(estimator, X_val, y_val).logloss
    @test training_loss_before_train > training_loss_after_train
    @test val_loss_before_train > val_loss_after_train
    # Sampling
    y_sampled = sample_from(estimator, X)
    @test y_sampled isa Vector
    # Expected_value
    oracle_prediction = sinusoid_function(X.x)
    network_prediction = TMLE.expected_value(estimator, X)
    network_mse = mean((network_prediction .- y).^2)
    oracle_mse = mean((oracle_prediction .- y).^2)
    relative_error = 100(network_mse - oracle_mse) / oracle_mse
    @test relative_error < 3 # less than 3 % difference between oracle and network
    # Check the mean is in bounds
    lb_μy, ub_μy = mean(y) - 1.96std(y), mean(y) + 1.96std(y)
    @test lb_μy <= mean(y_sampled) <= ub_μy
    # Check the variance is in bounds
    Χ² = Chisq(n_samples-1)
    lb_σy, ub_σy = (n_samples-1)*var(y)/quantile(Χ², 0.975), (n_samples-1)*var(y)/quantile(Χ², 0.025)
    @test lb_σy <= var(y_sampled) <= ub_σy
    # misc
    @test PopGenEstimatorComparison.infer_input_size(estimator.model) == 1
    @test PopGenEstimatorComparison.infer_K(estimator.model) == 3
    @test size(estimator.model.p_k_x[1][1].weight, 1) == 20
    new_model = PopGenEstimatorComparison.newmodel(estimator.model, (40,))
    @test size(new_model.p_k_x[1][1].weight, 1) == 40
end

@testset "Test SieveNeuralNetworkEstimator MixtureDensityNetwork" begin
    rng = Random.default_rng()
    Random.seed!(rng, 0)
    n_samples = 1000
    dataset = Float32.(sinusoidal_dataset(;n_samples=n_samples))
    X, y = X_y(dataset, [:x], :y)
    hidden_sizes_candidates = [(10,), (20,), (30,),]
    snne = SieveNeuralNetworkEstimator(X, y; 
        hidden_sizes_candidates=hidden_sizes_candidates,
        max_epochs=1000
    )
    snne, info = train!(snne, X, y;verbosity=0)
    # Last model has been selected
    @test size(snne.neural_net_estimator.model.p_k_x[1][1].weight, 1) == 30
    # Checking function have been implemented
    @test sample_from(snne, X) isa Vector
    @test evaluation_metrics(snne, X, y) isa NamedTuple{(:logloss,)}
    @test TMLE.expected_value(snne, X) isa Vector
end

@testset "Test CategoricalMLP" begin
    rng = Random.default_rng()
    Random.seed!(rng, 0)
    n = 1000
    X, y = MLJBase.make_circles(n)
    y = categorical(y, ordered=true)
    X = Float32.(DataFrame(X))
    X_train, y_train, X_val, y_val = train_validation_split(X, y)
    # Training and Evaluation
    estimator = NeuralNetworkEstimator(X, y, max_epochs=10_000)
    training_loss_before_train = evaluation_metrics(estimator, X_train, y_train).logloss
    val_loss_before_train = evaluation_metrics(estimator, X_val, y_val).logloss
    train!(estimator, X, y, verbosity=0)
    training_loss_after_train = evaluation_metrics(estimator, X_train, y_train).logloss
    val_loss_after_train = evaluation_metrics(estimator, X_val, y_val).logloss
    @test training_loss_before_train > training_loss_after_train
    @test val_loss_before_train > val_loss_after_train
    # Sample, it is almost perfect here because the problem is too simple
    y_sampled = sample_from(estimator, X)
    @test sum(y_sampled != y) < 5
    @test y_sampled isa CategoricalArrays.CategoricalVector
    labels = levels(y)
    @test levels(y_sampled) == labels
    # Expected_value    
    network_prediction = TMLE.expected_value(estimator, X)
    # No prediction closer to the incorect label
    @test sum(abs.(network_prediction .- float(y)) .> 0.5) == 0
    # misc
    @test PopGenEstimatorComparison.infer_input_size(estimator.model) == 2
    @test size(estimator.model[1][1].weight, 1) == 20
    new_model = PopGenEstimatorComparison.newmodel(estimator.model, (40,))
    @test size(new_model[1][1].weight, 1) == 40
end

@testset "Test SieveNeuralNetworkEstimator CategoricalMLP" begin
    rng = Random.default_rng()
    Random.seed!(rng, 0)
    n = 1000
    X, y = MLJBase.make_circles(n)
    y = categorical(y, ordered=true)
    X = Float32.(DataFrame(X))

    hidden_sizes_candidates = [(10,), (20,), (30,),]
    snne = SieveNeuralNetworkEstimator(X, y; 
        hidden_sizes_candidates=hidden_sizes_candidates,
        max_epochs=1000
    )
    snne, info = train!(snne, X, y;verbosity=0)
    # Last model has been selected
    @test size(snne.neural_net_estimator.model[1][1].weight, 1) == 30
    # Checking function have been implemented
    @test sample_from(snne, X) isa CategoricalVector
    @test evaluation_metrics(snne, X, y) isa NamedTuple{(:logloss,)}
    @test TMLE.expected_value(snne, X) isa Vector
end

end
true