using DataFrames
using Arrow
using CSV
using MLJ
using PopGenEstimatorComparison
using TMLE
using CairoMakie
using StatsBase

function get_model_distributions(Ψ, dataset; verbosity=0)
    dataset = shuffle(dataset)
    relevant_factors = TMLE.get_relevant_factors(Ψ)
    cache_all = Dict()
    model_keys = tuple(Ψ.outcome, keys(Ψ.treatment_values)...)
    model_combinations = [
        "All GLMNets" => NamedTuple{model_keys}(
        [
            PopGenEstimatorComparison.outcome_model(all_models, Ψ.outcome, dataset; flavor=:GLMNet), 
            PopGenEstimatorComparison.treatment_model(all_models, flavor=:GLMNet),
            PopGenEstimatorComparison.treatment_model(all_models, flavor=:GLMNet)
        ]
        ),
        "GLMNet/GLMNet/GLM" => NamedTuple{model_keys}(
            [
                PopGenEstimatorComparison.outcome_model(all_models, Ψ.outcome, dataset; flavor=:GLMNet), 
                PopGenEstimatorComparison.treatment_model(all_models, flavor=:GLMNet),
                PopGenEstimatorComparison.treatment_model(all_models, flavor=:GLM)
            ]
        ),
        "GLMNet/GLM/GLMNet" => NamedTuple{model_keys}(
            [
                PopGenEstimatorComparison.outcome_model(all_models, Ψ.outcome, dataset; flavor=:GLMNet), 
                PopGenEstimatorComparison.treatment_model(all_models, flavor=:GLM),
                PopGenEstimatorComparison.treatment_model(all_models, flavor=:GLMNet)
            ]
        ),
        "GLM/GLMNet/GLMNet" => NamedTuple{model_keys}(
            [
                PopGenEstimatorComparison.outcome_model(all_models, Ψ.outcome, dataset; flavor=:GLM), 
                PopGenEstimatorComparison.treatment_model(all_models, flavor=:GLMNet),
                PopGenEstimatorComparison.treatment_model(all_models, flavor=:GLMNet)
            ]
        ),
        "All GLMs" => NamedTuple{model_keys}(
            [
                PopGenEstimatorComparison.outcome_model(all_models, Ψ.outcome, dataset; flavor=:GLM), 
                PopGenEstimatorComparison.treatment_model(all_models, flavor=:GLM),
                PopGenEstimatorComparison.treatment_model(all_models, flavor=:GLM)
            ]
        )
    ]
    results = []
    for (model_comb, models) in model_combinations
        tmle = TMLEE(models, weighted=false)
        result_all, cache_all = tmle(Ψ, dataset, cache=cache_all, verbosity=verbosity);
        # Q init
        Qinit = cache_all[relevant_factors.outcome_mean][2]
        ŷinit = MLJ.predict(Qinit.machine)
        # Q star
        Qstar = cache_all[:last_fluctuation].outcome_mean
        ŷstar = MLJ.predict(Qstar.machine)
        # G
        G1, G2 = cache_all[:last_fluctuation].propensity_score
        g1 = MLJ.predict(G1.machine)
        g2 = MLJ.predict(G2.machine)
        push!(results, model_comb => (initial=ŷinit, targeted=ŷstar, tmle=result_all, g1=g1, g2=g2, qstar=Qstar))
    end
    return results
end

function plot_model_distributions(results)
    xs = 1:length(results)
    xticks=(xs, [x[1] for x ∈ results])
    fig = Figure(resolution=(1000, 1000))
    # Q⋆
    ax1 = Axis(fig[1, 1],
        title="Q⋆ Distribution",
        yticks=xticks,
        limits=((0, 0.01), nothing)
    )
    for (model_index, (_, result)) ∈ enumerate(results)
        μs = TMLE.expected_value(result.targeted)
        hist!(ax1, μs,
            bins=1000,
            scale_to=0.7, 
            offset=model_index,
        )
    end
    # Confidence Intervals
    ax2 =  Axis(fig[2, 1], 
        title="Confidence intervals",
        xlabel="Effect size",
        yticks=xticks)
    estimates = [x[2].tmle.Ψ̂ for x in results]
    upbs = [confint(OneSampleTTest(x[2].tmle))[2] for x in results]
    errorbars!(ax2, estimates, xs, upbs .- estimates, 
        whiskerwidth = 10,
        direction=:x,
        linewidth=2)
    scatter!(ax2, estimates, xs, markersize=10, color=:black)
    vlines!(ax2, 0, color=:green)
    # G₁
    distributions = ["G₁" => :g1, "G₂" => :g2]
    glmnet_results = results[1]
    glm_results = results[end]
    @assert glmnet_results[1] == "All GLMNets"
    @assert glm_results[1] == "All GLMs"
    colors = Makie.wong_colors();
    for (ax_id, (distr_name, distr_symbol)) in enumerate(distributions)
        ax =  Axis(fig[ax_id, 2], 
            title=string(distr_name, " Distribution"),
            yticks=(1:2, ["GLMNet", "GLM"]),
        )
        distr_glmnet = glmnet_results[2][distr_symbol]
        distr_glm = glm_results[2][distr_symbol]
        for (model_index, distr) in enumerate((distr_glmnet, distr_glm))
            for (class_id, class) in enumerate(sort(classes(distr)))
                μs = pdf.(distr, class)
                hist!(ax, μs,
                    bins=100,
                    scale_to=0.7,
                    label=string(class),
                    color=colors[class_id],
                    offset=model_index, 
                )
            end
        end
        axislegend(merge=true)
    end
    return fig
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
dataset = Arrow.Table(datasets["All"]) |> DataFrame

# Estimand
id = 6
row = problematic_estimands[id, :]
Ψ = PopGenEstimatorComparison.estimand_from_results_row(row)

# Models distribution comparison
PopGenEstimatorComparison.coercetypes!(dataset, Ψ)
dataset = dropmissing(dataset, vcat(keys(Ψ.treatment_values)..., Ψ.outcome, Ψ.outcome_extra_covariates...))
results = get_model_distributions(Ψ, dataset; verbosity=0)
plot_model_distributions(results)

# Plot Q⋆ output
reldistance_threshold = 0.01
ystar_glmnets = TMLE.expected_value(results[1][2].targeted)
ystar_mix = TMLE.expected_value(results[3][2].targeted)
fig = Figure()
ax = Axis(fig[1, 1], title="Q⋆ predictions", xlabel="GLMNets", ylabel="GLMNet/GLM/GLMNets")
scatter!(ax, ystar_glmnets, ystar_mix, label="All")
ablines!(ax, 0, 1, color=:green)
outliers_ids = [index for (index, (y1, y2)) in enumerate(zip(ystar_glmnets, ystar_mix)) if abs(y1 - y2)/y2 > reldistance_threshold]
scatter!(ax, ystar_glmnets[outliers_ids], ystar_mix[outliers_ids], color=:red, label="Outliers (τ=$reldistance_threshold)")
axislegend(ax)
fig

dataset_no_outliers = dataset[Not(outliers_ids), :]
results_no_outliers = get_model_distributions(Ψ, dataset_no_outliers; verbosity=0)
plot_model_distributions(results_no_outliers)

qstar_glmnets = results_no_outliers[1][2].qstar
qstar_mix = results_no_outliers[3][2].qstar

## Covariate analysis

outlier_indices = []
dataset_no_outliers = dataset[Not(outlier_indices), :]

cache = Dict()
tmle = TMLEE(model_combinations[1][2], weighted=false)
result_glmnets, cache = tmle(Ψ, dataset_no_outliers, cache=cache, verbosity=1);
result_glmnets
Qstar_glmnets = cache[:last_fluctuation].outcome_mean
ŷstar_glmnets = MLJ.predict(Qstar_glmnets.machine)

tmle = TMLEE(model_combinations[3][2], weighted=false)
result_mix, cache = tmle(Ψ, dataset_no_outliers, cache=cache, verbosity=1);
result_mix
Qstar_mix = cache[:last_fluctuation].outcome_mean
ŷstar_mix = MLJ.predict(Qstar_mix.machine)

fig = Figure()
ax = Axis(fig[1, 1],
    title = "Clever covariate comparison",
    xlabel = "GLMnets",
    ylabel = "GLMnet/GLM/GLMNet"
)
scatter!(ax, 
    Qstar_glmnets.machine.fitresult.one_dimensional_path.data[1].covariate, 
    Qstar_mix.machine.fitresult.one_dimensional_path.data[1].covariate
)
ablines!(ax, 0, 1, label="Identity")
axislegend()
fig

covariate_outliers = DataFrame(
    baseline = Qstar_glmnets.machine.fitresult.one_dimensional_path.data[1].covariate, 
    problem = Qstar_mix.machine.fitresult.one_dimensional_path.data[1].covariate
)
covariate_outliers.reldiff = abs.(covariate_outliers.baseline .- covariate_outliers.problem)
outlier_indices = findall(covariate_outliers.reldiff .> 5)

covariate_outliers[outlier_indices, :]