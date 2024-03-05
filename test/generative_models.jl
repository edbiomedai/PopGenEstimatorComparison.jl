module TestGenerativeModels

using PopGenEstimatorComparison
using PopGenEstimatorComparison: logloss, early_stopping_message
using Test
using Random
using DataFrames
using TMLE
using Distributions
using MLJBase

function sinusoidal_dataset(;n_samples=100)
    ϵ = rand(Normal(), n_samples)
    x = rand(Uniform(-10.5, 10.5), n_samples)
    y = 7sin.(0.75x) .+ 0.5x .+ ϵ
    return DataFrame(x=x, y=y)
end

@testset "Test RandomDatasetGenerator" begin
    generator = RandomDatasetGenerator()
    sample_size = 10
    dataset = sample(generator, sample_size)
    @test names(dataset) == ["W_1", "W_2", "W_3", "W_4", "W_5", "W_6", "T_1", "Y"]
    @test size(dataset) == (sample_size, length(names(dataset)))
end

@testset "Test MixtureDensityNetwork" begin
    # Seeding
    rng = Random.default_rng()
    Random.seed!(rng, 0)
    dataset = Float32.(sinusoidal_dataset(;n_samples=1000))
    X, y = DataFrame(x=dataset.x,), dataset.y
    # Fitting
    estimator = MixtureDensityNetworkEstimator()
    Xmat, ymat = PopGenEstimatorComparison.reformat(X, y)
    loss_before_train = logloss(estimator.model(Xmat, ymat))
    @test_logs (:info, early_stopping_message(5)) train!(estimator, X, y, verbosity=0)
    loss_after_train = logloss(estimator.model(Xmat, ymat))
    @test loss_before_train > loss_after_train
    # Sampling
    PopGenEstimatorComparison.sample(estimator, X)

end

end

true