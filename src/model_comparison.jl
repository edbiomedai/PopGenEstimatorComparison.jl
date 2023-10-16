using DataFrames
using Arrow
using CSV
using MLJ
using PopGenEstimatorComparison
using TMLE
using CairoMakie

function evaluate_models(models, Ψ, all_individuals; verbosity=0)
    relevant_factors = TMLE.get_relevant_factors(Ψ)
    cache_all = Dict()
    # Only GLMnets
    model_combinations = [
        "All GLMNets" => NamedTuple{model_keys}(
        [
            PopGenEstimatorComparison.outcome_model(all_models, Ψ.outcome, all_individuals; flavor=:GLMNet), 
            PopGenEstimatorComparison.treatment_model(all_models, flavor=:GLMNet),
            PopGenEstimatorComparison.treatment_model(all_models, flavor=:GLMNet)
        ]
    ),
    "GLMNet/GLMNet/GLM" => NamedTuple{model_keys}(
        [
            PopGenEstimatorComparison.outcome_model(all_models, Ψ.outcome, all_individuals; flavor=:GLMNet), 
            PopGenEstimatorComparison.treatment_model(all_models, flavor=:GLMNet),
            PopGenEstimatorComparison.treatment_model(all_models, flavor=:GLM)
        ]
    ),
    "GLMNet/GLM/GLMNet" => NamedTuple{model_keys}(
        [
            PopGenEstimatorComparison.outcome_model(all_models, Ψ.outcome, all_individuals; flavor=:GLMNet), 
            PopGenEstimatorComparison.treatment_model(all_models, flavor=:GLM),
            PopGenEstimatorComparison.treatment_model(all_models, flavor=:GLMNet)
        ]
    ),
    "GLM/GLMNet/GLMNet" => NamedTuple{model_keys}(
        [
            PopGenEstimatorComparison.outcome_model(all_models, Ψ.outcome, all_individuals; flavor=:GLM), 
            PopGenEstimatorComparison.treatment_model(all_models, flavor=:GLMNet),
            PopGenEstimatorComparison.treatment_model(all_models, flavor=:GLMNet)
        ]
    ),
    "All GLMs" => NamedTuple{model_keys}(
        [
            PopGenEstimatorComparison.outcome_model(all_models, Ψ.outcome, all_individuals; flavor=:GLM), 
            PopGenEstimatorComparison.treatment_model(all_models, flavor=:GLM),
            PopGenEstimatorComparison.treatment_model(all_models, flavor=:GLM)
        ]
    )
    ]
    fig = Figure()
    ax = Axis(fig[1, 1], title=model_comb)
    for (index, (model_comb, models)) in enumerate(model_combinations)
        tmle = TMLEE(models, weighted=false)
        result_all, cache_all = tmle(Ψ, all_individuals, cache=cache_all, verbosity=verbosity);
        Qinit = cache_all[relevant_factors.outcome_mean][2]
        ŷinit = log.(TMLE.expected_value(predict(Qinit.machine)))
        init = hist!(ax, ŷinit, bins=100, scale_to=-0.6, label="Initial", offset=index, normalization=:pdf, direction=:x)
        Qstar = cache_all[:last_fluctuation].outcome_mean
        ŷstar = log.(TMLE.expected_value(predict(Qstar.machine)))
        targeted = hist!(ax, ŷstar, bins=100, scale_to=-0.6, label="Targeted", offset=index, normalization=:pdf, direction=:x)
    end

    save("test.png", fig)

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
models = [all_models[:GLM_CATEGORICAL], all_models[:GLMNet_CATEGORICAL]]
