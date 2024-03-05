### Random Generator

struct RandomDatasetGenerator
    confounders_distribution
    treatments_distribution
    outcome_distribution
end

function RandomDatasetGenerator(;
    confounders_distribution=MvNormal(ones(6), Diagonal(ones(6))),
    treatments_distribution=Bernoulli(0.01),
    outcome_distribution=Bernoulli(0.01)
    )
    return RandomDatasetGenerator(
        confounders_distribution, 
        treatments_distribution, 
        outcome_distribution
    )
end

maybe_transpose(x::AbstractVector) = reshape(x, length(x), 1)
maybe_transpose(x) = collect(transpose(x))

function sample_from_distribution(d::Distribution, n; rng=Random.default_rng(), prefix=:V)
    samples = maybe_transpose(rand(rng, d, n))
    dim = size(samples, 2)
    colnames = [Symbol(prefix, i) for i in 1:dim]
    return DataFrame(samples, colnames)
end

function sample_from_distribution(ds, n; rng=Random.default_rng(), prefix=:V)
    samples = [sample_from_distribution(d, n; rng=rng, prefix=Symbol(prefix, index)) for (index, d) in enumerate(ds)]
    return hcat(samples...)
end

function sample(generator::RandomDatasetGenerator, n::Int; rng=Random.default_rng())
    W = sample_from_distribution(generator.confounders_distribution, n; rng=rng, prefix=:W_)
    T = sample_from_distribution(generator.treatments_distribution, n;rng=rng, prefix=:T_)
    Y = rand(rng, generator.outcome_distribution, n)
    dataset = hcat(W, T)
    dataset.Y = Y
    return dataset
end

### Mixture Density Network

"""
Implements a Mixture of Gaussians Density Network.

Attributes:

- p_k_x: conditional probability of k giuven x
- σ_xk: standard deviation of the k-th gaussian as a function of x
- μ_xk: mean of the k-th gaussian as a function of x

"""
struct MixtureDensityNetwork
    p_k_x::Chain
    σ_xk::Chain
    μ_xk::Chain
end

function MixtureDensityNetwork(;n_inputs=1, n_hidden=20, K=3)
    z_h = Dense(n_inputs, n_hidden, tanh)
    z_k = Dense(n_hidden, K)
    z_σ = Dense(n_hidden, K, exp)
    z_μ = Dense(n_hidden, K)
    p_k_x = Chain(z_h, z_k, softmax)
    σ_xk = Chain(z_h, z_σ)
    μ_xk = Chain(z_h, z_μ)
    return MixtureDensityNetwork(p_k_x, σ_xk, μ_xk)
end

function (model::MixtureDensityNetwork)(x, y)
    p_k_x = model.p_k_x(x)
    σ_xk = model.σ_xk(x)
    μ_xk = model.μ_xk(x)
    l = p_k_x.*gaussian_kernel(y, μ_xk, σ_xk)
    return sum(l, dims=1)
end

Flux.@functor MixtureDensityNetwork

reformat(vector::AbstractVector) = reshape(vector, 1, length(vector))
reformat(table) = collect(transpose(MLJBase.matrix(table)))
reformat(table, vector::AbstractVector) = reformat(table), reformat(vector)

function reformat(X, y, rows)
    X_rows = selectrows(X, rows)
    y_rows = selectrows(y, rows)
    return reformat(X_rows, y_rows)
end

gaussian_kernel(y, μ, σ) = 1 ./ ((sqrt(2π).*σ)) .* exp.(-0.5((y .- μ)./σ).^2)

logloss(x) = - mean(log.(x))

early_stopping_message(patience) =  string("Validation Loss stopped decreasing for ", patience, " consecutive epochs. Early Stopping.")

function train!(estimator, X, y; verbosity=1)
    model = estimator.model

    # Dataloaders
    n = length(y)
    train_rows, val_rows = first(MLJBase.train_test_pairs(estimator.resampling, 1:n, X, y))
    X_train, y_train = reformat(X, y, train_rows)
    X_val, y_val = reformat(X, y, val_rows)
    n_train, n_val = length(y_train), length(y_val)
    train_loader = Flux.DataLoader((X_train, y_train), batchsize=estimator.batchsize)
    val_loader = Flux.DataLoader((X_val, y_val), batchsize=estimator.batchsize)
    
    # Training Loop
    local epoch_validation_loss = Inf
    es = Flux.early_stopping(() -> epoch_validation_loss, estimator.patience; init_score = Inf)
    opt_state = Flux.setup(estimator.optimiser, model)
    for epoch in 1:estimator.max_epochs
        epoch_training_loss = 0.
        for (xbatch, ybatch) in train_loader
            training_loss, grads = Flux.withgradient(model) do m
                logloss(m(xbatch, ybatch))
            end
            epoch_training_loss += training_loss*length(ybatch)
            Flux.update!(opt_state, model, grads[1])
        end
        epoch_training_loss /= n_train
        epoch_validation_loss = sum(length(ybatch)*logloss(model(xbatch, ybatch)) for (xbatch, ybatch) in val_loader) / n_val
        if verbosity > 0 
            @info string("Epoch: ", epoch, ", Training Loss: ",  epoch_training_loss, ", Validation Loss: ", epoch_validation_loss)
        end
        if es()
            @info early_stopping_message(estimator.patience)
            break
        end
    end
    return 
end

struct MixtureDensityNetworkEstimator
    model::MixtureDensityNetwork
    optimiser::Flux.Optimise.AbstractOptimiser
    resampling::MLJBase.ResamplingStrategy
    batchsize::Int
    patience::Int
    max_epochs::Int
end

MixtureDensityNetworkEstimator(;
    model = MixtureDensityNetwork(),
    optimiser = Adam(),
    resampling = Holdout(),
    batchsize = 64,
    patience = 5,
    max_epochs = 10_000,
    ) = MixtureDensityNetworkEstimator(model, optimiser, resampling, batchsize, patience, max_epochs)

function gumbel_sample(x)
    z = rand(Gumbel(), size(x))
    return argmax(log.(x) + z, dims=1)
end

function sample(model::MixtureDensityNetwork, X)
    X = reformat(X)
    k_x = gumbel_sample(model.p_k_x(X))
    σ_xk = model.σ_xk(X)
    μ_xk = model.μ_xk(X)
    return rand(Normal(), 1, length(k_x)) .* σ_xk[k_x] + μ_xk[k_x]
end

sample(estimator, X) = sample(estimator.model, X)

### Categorical MLP

### General Density Estimation Procedure

function autoregressive_density_estimation(dataset, outcome, treatments; 
    confounders=("PC1", "PC2", "PC3", "PC4", "PC5", "PC6"), 
    outcome_extra_covariates=("Age-Assessment", "Genetic-Sex");
    )
    outcome_distribution = TMLE.ConditionalDistribution(outcome, vcat(treatments..., confounders..., outcome_extra_covariates...))
    outcome_estimator = MixtureDensityNetworkEstimator(model=MixtureDensityNetwork(n_inputs=length(outcome_distribution.parents)))
    outcome_dataset = dropmissing(dataset, [outcome_distribution.parents..., outcome_distribution.outcome])
    X = outcome_dataset[!, outcome_distribution.parents]
    y = outcome_dataset[!, outcome_distribution.outcome]
    train!(outcome_estimator, X, y)
    treatment_distributions = [TMLE.ConditionalDistribution(treatment, confounders) for treatment in treatments]

end