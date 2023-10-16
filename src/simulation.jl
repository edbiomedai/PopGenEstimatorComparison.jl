using Lux
using CSV
using Arrow
using DataFrames
using TMLE
using Random

# Data
dataset = Arrow.Table(datasets["All"]) |> DataFrame

# Parameter of interest
problematic_estimands = CSV.read(estimands_file, DataFrame)
id = 6
row = problematic_estimands[id, :]
Ψ = PopGenEstimatorComparison.estimand_from_results_row(row)
relevant_factors = TMLE.get_relevant_factors(Ψ)


# Fit y | X
rng = MersenneTwister()
Random.seed!(rng, 123)
outcome_factor = relevant_factors.outcome_mean

data = dropmissing(dataset, vcat(collect(outcome_factor.parents), outcome_factor.outcome))

X = data[!, collect(outcome_factor.parents)]
y = 
model = Chain(Dense(1 => 16, relu), Dense(16 => 1))