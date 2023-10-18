using Lux
using CSV
using Arrow
using DataFrames
using TMLE
using Random
using PopGenEstimatorComparison
using MLJ
using MLJModels
using Zygote
using Optimisers
# using Metal
import AbstractDifferentiation as AD
import MLUtils: DataLoader, splitobs

function xlogy(x, y)
    result = x * log(y)
    return ifelse(iszero(x), zero(result), result)
end

function binarycrossentropy(y_pred, y_true)
    y_pred = y_pred .+ eps(eltype(y_pred))
    return mean(@. -xlogy(y_true, y_pred) - xlogy(1 - y_true, 1 - y_pred))
end

function compute_loss(model, ps, st, data)
    x, y = data
    ŷ, st = model(x, ps, st)
    return binarycrossentropy(ŷ, y), st, ()
end

# Data
datasets = Dict(
    "All" => "data/all_population_data.arrow",
    "White" => "data/white_population_data.arrow",
)
dataset = Arrow.Table(datasets["All"]) |> DataFrame

# Parameter of interest
estimands_file = "data/problematic_estimands.csv"
problematic_estimands = CSV.read(estimands_file, DataFrame)
id = 6
row = problematic_estimands[id, :]
Ψ = PopGenEstimatorComparison.estimand_from_results_row(row)
relevant_factors = TMLE.get_relevant_factors(Ψ)


# Fit y | X
rng = MersenneTwister()
Random.seed!(rng, 123)
batchsize = 16
outcome_factor = relevant_factors.outcome_mean

data = dropmissing(dataset, vcat(collect(outcome_factor.parents), outcome_factor.outcome))
X = data[!, collect(outcome_factor.parents)]
treatments = collect(keys(Ψ.treatment_values))
for treatment in treatments
    X[!, treatment] = categorical(X[!, treatment])
end
onehot_mach = machine(OneHotEncoder(drop_last=true), X)
fit!(onehot_mach, verbosity=0)
Xt = convert(Matrix{Float32}, MLJ.matrix(MLJ.transform(onehot_mach, X), transpose=true))
y = convert(Vector{Float32}, data[!, outcome_factor.outcome])

(X_train, y_train), (X_test, y_test) = splitobs((Xt, y); at=0.7)

train_dataloader = DataLoader(collect.((X_train, y_train)); batchsize, shuffle=true)
test_dataloader = DataLoader(collect.((X_test, y_test)); batchsize, shuffle=false)

model = Chain(Dense(12 => 16, relu), Dense(16 => 16, relu), Dense(16 => 1, sigmoid))
dev = cpu_device()
opt = Adam()
tstate = Lux.Training.TrainState(rng, model, opt)
vjp = Lux.Training.AutoZygote()

function main(tstate::Lux.Experimental.TrainState, vjp, train_dataloader, test_dataloader, epochs)
    for epoch in 1:epochs
        trainloss = 0.
        for batch in train_dataloader
            grads, loss, stats, tstate = Lux.Training.compute_gradients(vjp,
                compute_loss, batch, tstate)
                trainloss += loss
            tstate = Lux.Training.apply_gradients(tstate, grads)
        end
        trainloss /= size(train_dataloader.data[1], 2)
        println("Epoch: $(epoch) || Training Loss: $(trainloss)")

        ytest = categorical(test_dataloader.data[2])
        ŷ = model(test_dataloader.data[1], tstate.parameters, tstate.states)[1]
        ŷ = UnivariateFinite(categorical([0, 1]), ŷ[1, :], augment=true)
        valloss = mean(log_loss(ŷ, ytest))
        println("Epoch: $(epoch) || Validation Loss: $(valloss)")
    end
    return tstate
end

tstate = main(tstate, vjp, train_dataloader, test_dataloader, 100)

