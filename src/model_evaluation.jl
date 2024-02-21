using DataFrames
using Arrow
using CSV
using MLJ
using PopGenEstimatorComparison
using TMLE
using CairoMakie

function evaluate_models(models, relevant_factors, all_individuals; nfolds=3, acceleration=CPU1())
    results = []
    # Outcome models evaluation
    outcome_factor = relevant_factors.outcome_mean
    data = dropmissing(all_individuals, vcat(outcome_factor.parents..., outcome_factor.outcome))
    X = data[!, collect(outcome_factor.parents)]
    y = data[!, outcome_factor.outcome]
    outcome_results = []
    for model in models
        push!(
            outcome_results, 
            evaluate(with_encoder(model), X, y, resampling=StratifiedCV(nfolds=nfolds), measure=log_loss, acceleration=acceleration)
        )
    end
    push!(results, outcome_results)

    for ps_factor ∈ relevant_factors.propensity_score
        data = dropmissing(all_individuals, vcat(ps_factor.parents..., ps_factor.outcome))
        X = data[!, collect(ps_factor.parents)]
        y = data[!, ps_factor.outcome]
        ps_factor_results = []
        for model in models
            push!(
                ps_factor_results, 
                evaluate(model, X, y, resampling=StratifiedCV(nfolds=nfolds), measure=log_loss, acceleration=acceleration)
            )
        end
        push!(results, ps_factor_results)
    end
    return results
end

# Constants
datasets = Dict(
    "All" => "data/all_population_data.arrow",
    "White" => "data/white_population_data.arrow",
)
estimands_file = "data/problematic_estimands.csv"
verbosity = 1

# Load Data
problematic_estimands = CSV.read(estimands_file, DataFrame)
all_individuals = Arrow.Table(datasets["All"]) |> DataFrame

# Estimand
id = 6
row = problematic_estimands[id, :]
Ψ = PopGenEstimatorComparison.estimand_from_results_row(row)


# Models evaluation
PopGenEstimatorComparison.coercetypes!(all_individuals, Ψ)
relevant_factors = TMLE.get_relevant_factors(Ψ)
models = [all_models[:GLM_CATEGORICAL], all_models[:GLMNet_CATEGORICAL]]

nfolds = 10
nmodels = length(models)
results = evaluate_models(models, relevant_factors, all_individuals; nfolds=nfolds, acceleration=CPUThreads())

fig = Figure()
for (factor_index, factor_results) in enumerate(results)
    ax = Axis(fig[factor_index, 1], aspect = AxisAspect(2), xticks=(1:nmodels, [string(Base.typename(typeof(models[i])).name) for i in 1:nmodels]))
    losses = [factor_result.measurement[1] for factor_result in factor_results]
    stdes = [1.96*sqrt(var(factor_results[1].per_fold[1])/nfolds) for factor_result in factor_results]
    errorbars!(ax, 1:nmodels, losses, stdes,
        color = range(0, 1, length = nmodels),
        whiskerwidth = 10)
end
fig
