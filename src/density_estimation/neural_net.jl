### Neural Network Estimator

early_stopping_message(patience) =  string("Validation Loss stopped decreasing for ", patience, " consecutive epochs. Early Stopping.")

maybe_float_32(x::AbstractVector{<:AbstractFloat}) = convert(Vector{Float32}, x)
maybe_float_32(x) = x

function X_y(dataset, Xcols, ycol)
    outcome_dataset = dropmissing(dataset, Symbol.([Xcols..., ycol]))
    for colname in names(outcome_dataset)
        outcome_dataset[!, colname] = maybe_float_32(outcome_dataset[!, colname])
    end
    X = outcome_dataset[:, collect(Xcols)]
    y = outcome_dataset[:, ycol]
    return X, y
end

function train!(estimator, X, y; verbosity=1)
    model = estimator.model

    # Dataloaders
    X_train, y_train, X_val, y_val = net_train_validation_split(estimator.resampling, X, y)
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
                PopGenEstimatorComparison.compute_loss(m, xbatch, ybatch)
            end
            epoch_training_loss += training_loss*length(ybatch)
            Flux.update!(opt_state, model, grads[1])
        end
        epoch_training_loss /= n_train
        epoch_validation_loss = sum(length(ybatch)*compute_loss(model, xbatch, ybatch) for (xbatch, ybatch) in val_loader) / n_val
        if verbosity > 1 
            @info string("Epoch: ", epoch, ", Training Loss: ",  epoch_training_loss, ", Validation Loss: ", epoch_validation_loss)
        end
        if epoch_validation_loss == 0 || es()
            verbosity > 0 && @info early_stopping_message(estimator.patience)
            break
        end
    end
    return 
end

struct NeuralNetworkEstimator
    model
    optimiser::Flux.Optimise.AbstractOptimiser
    resampling::MLJBase.ResamplingStrategy
    batchsize::Int
    patience::Int
    max_epochs::Int
end

NeuralNetworkEstimator(model;
    optimiser = Adam(),
    resampling = Holdout(),
    batchsize = 64,
    patience = 5,
    max_epochs = 10,
    ) = NeuralNetworkEstimator(model, optimiser, resampling, batchsize, patience, max_epochs)

NeuralNetworkEstimator(X, y::AbstractVector{<:AbstractFloat}, hidden_sizes; K=3, kwargs...) = 
    NeuralNetworkEstimator(MixtureDensityNetwork(
            input_size=get_input_size(X), 
            hidden_sizes=hidden_sizes,
            K=K
            );
        kwargs...
    )

function NeuralNetworkEstimator(X, y::CategoricalVector, hidden_sizes; K=3, kwargs...)
    input_size = get_input_size(X)
    output_size = length(levels(y))
    hidden_sizes = tuple(hidden_sizes..., output_size)
    NeuralNetworkEstimator(CategoricalMLP(
            input_size=input_size, 
            hidden_sizes=hidden_sizes,
        );
        kwargs...
    )
end

sample_from(estimator::NeuralNetworkEstimator, X::DataFrame, labels=nothing) = 
    sample_from(estimator.model, encode_or_reformat(X), labels)

function evaluation_metrics(estimator::NeuralNetworkEstimator, X, y)
    ŷ = estimator.model(encode_or_reformat(X))
    encode_or_reformat(ŷ)
    Flux.crossentropy(model(x), ŷ)
end

### Categorical MLP

"""

Creates a sequence of hidden layers.
Note: The last hidden layer has no activation function
"""
function hidden_layers(;input_size=1, hidden_sizes=(32,), activation=relu)
    length(hidden_sizes) == 1 && return Dense(input_size, hidden_sizes[1])
    last = Dense(hidden_sizes[end-1], hidden_sizes[end])
    ins = vcat(input_size, hidden_sizes[1:end-2]...)
    outs = collect(hidden_sizes[1:end-1])
    return Chain((Dense(in, out, activation) for (in, out) in zip(ins, outs))..., last)
end

CategoricalMLP(;input_size=1, hidden_sizes=(32, 2)) =
    hidden_layers(input_size=input_size, hidden_sizes=hidden_sizes)

compute_loss(model, x, y::OneHotMatrix) = Flux.logitcrossentropy(model(x), y)

function sample_from(model::Chain, X, labels)
    ps = softmax(model(X))
    indicators = [rand(Categorical(collect(p))) for p in eachcol(ps)]
    return categorical([labels[i] for i in indicators], levels=labels)
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

function MixtureDensityNetwork(;input_size=1, hidden_sizes=(20,), K=3)
    last_hidden_size = hidden_sizes[end]
    z_h = Chain(hidden_layers(input_size=input_size, hidden_sizes=hidden_sizes), x -> tanh.(x))
    z_k = Dense(last_hidden_size, K)
    z_σ = Dense(last_hidden_size, K, exp)
    z_μ = Dense(last_hidden_size, K)
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

gaussian_kernel(y, μ, σ) = 1 ./ ((sqrt(2π).*σ)) .* exp.(-0.5((y .- μ)./σ).^2)

logloss(x) = - mean(log.(x))

compute_loss(model, x, y::Matrix{<:AbstractFloat}) = logloss(model(x, y))

function gumbel_sample(x)
    z = rand(Gumbel(), size(x))
    return argmax(log.(x) + z, dims=1)
end

function sample_from(model::MixtureDensityNetwork, X, labels=nothing)
    k_x = gumbel_sample(model.p_k_x(X))
    σ_xk = model.σ_xk(X)
    μ_xk = model.μ_xk(X)
    return rand(Normal(), length(k_x)) .* vec(σ_xk[k_x]) .+ vec(μ_xk[k_x])
end