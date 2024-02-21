using CSV
using DataFrames
using Arrow
using PopGenEstimatorComparison
using MLJXGBoostInterface
using TMLE
using MLJ
using Random
using CairoMakie
using MLJLinearModels
using MLJModels

function simulate_new_outcomes(dataset, Ψ; 
    ymodel=with_encoder(XGBoostClassifier(num_round=1000, seed=123)), 
    rng=MersenneTwister(123)
    )
    Ψ_dataset = dropmissing(select(dataset, variables_from_estimand(Ψ)))
    PopGenEstimatorComparison.coercetypes!(Ψ_dataset, Ψ)
    outcome_mean = TMLE.get_relevant_factors(Ψ).outcome_mean
    X = Ψ_dataset[!, collect(outcome_mean.parents)]
    y = Ψ_dataset[!, outcome_mean.outcome]
    mach = machine(ymodel, X, y)
    fit!(mach, verbosity=0)
    Ψ_dataset[!, outcome_mean.outcome] = rand.(rng, predict(mach, X))
    return Ψ_dataset, mach
end

function plot_errorbars(results, approx_truth)
    estimates = [TMLE.estimate(result[2]) for result in results]
    upper_bounds = [confint(TMLE.OneSampleTTest(result[2]))[2] for result in results]
    nestimators = length(results)
    fig = Figure()
    ax = Axis(fig[1, 1], yticks = (1:nestimators, [x[1] for x ∈ results]))
    errorbars!(ax, 
        estimates,
        1:nestimators, 
        upper_bounds .- estimates,
        whiskerwidth = 15,
        color=1:nestimators,
        colormap=:seaborn_colorblind,
        colorrange = (1, nestimators),
        direction = :x
    )
    scatter!(ax, estimates, 1:nestimators, 
        markersize=15, 
        color=1:nestimators,
        colormap=:seaborn_colorblind,
        colorrange = (1, nestimators)
    )
    vlines!(ax, approx_truth)
    return fig
end

function approx_ATE(Ψ, sim_mach, sim_dataset)
    outcome_mean_estimate = TMLE.MLConditionalDistribution(
        TMLE.get_relevant_factors(Ψ).outcome_mean, 
        sim_mach
    )
    return mean(TMLE.counterfactual_aggregate(Ψ, outcome_mean_estimate, sim_dataset))
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
sim_model = with_encoder(XGBoostClassifier(num_round=1000, seed=123))

sim_dataset, sim_mach = simulate_new_outcomes(dataset, Ψ; 
    ymodel=sim_model, 
    rng=MersenneTwister(123)
    )

model_keys = tuple(Ψ.outcome, keys(Ψ.treatment_values)...)
model_combinations = [
    "All GLMNets" => NamedTuple{model_keys}(
    [
        PopGenEstimatorComparison.outcome_model(all_models, Ψ.outcome, dataset; flavor=:GLMNet), 
        PopGenEstimatorComparison.treatment_model(all_models, flavor=:GLMNet),
        PopGenEstimatorComparison.treatment_model(all_models, flavor=:GLMNet)
    ]),
    "All GLMs" => NamedTuple{model_keys}(
    [
        PopGenEstimatorComparison.outcome_model(all_models, Ψ.outcome, dataset; flavor=:GLM), 
        PopGenEstimatorComparison.treatment_model(all_models, flavor=:GLM),
        PopGenEstimatorComparison.treatment_model(all_models, flavor=:GLM)
    ]),
    "GLM/GLMnet/GLMnet" => NamedTuple{model_keys}(
    [
        PopGenEstimatorComparison.outcome_model(all_models, Ψ.outcome, dataset; flavor=:GLM), 
        PopGenEstimatorComparison.treatment_model(all_models, flavor=:GLMNet),
        PopGenEstimatorComparison.treatment_model(all_models, flavor=:GLMNet)
    ]),
    "GLMNet/GLM/GLMnet" => NamedTuple{model_keys}(
        [
            PopGenEstimatorComparison.outcome_model(all_models, Ψ.outcome, dataset; flavor=:GLMNet), 
            PopGenEstimatorComparison.treatment_model(all_models, flavor=:GLM),
            PopGenEstimatorComparison.treatment_model(all_models, flavor=:GLMNet)
        ]),
    "GLMNet/GLM/GLM    " => NamedTuple{model_keys}(
        [
            PopGenEstimatorComparison.outcome_model(all_models, Ψ.outcome, dataset; flavor=:GLMNet), 
            PopGenEstimatorComparison.treatment_model(all_models, flavor=:GLM),
            PopGenEstimatorComparison.treatment_model(all_models, flavor=:GLM)
        ]),
    "GLMNet/GLMNet/GLM" => NamedTuple{model_keys}(
        [
            PopGenEstimatorComparison.outcome_model(all_models, Ψ.outcome, dataset; flavor=:GLMNet), 
            PopGenEstimatorComparison.treatment_model(all_models, flavor=:GLMNet),
            PopGenEstimatorComparison.treatment_model(all_models, flavor=:GLM)
        ]),
    "Truth/GLM/GLM" => NamedTuple{model_keys}(
        [
            sim_model, 
            PopGenEstimatorComparison.treatment_model(all_models, flavor=:GLM),
            PopGenEstimatorComparison.treatment_model(all_models, flavor=:GLM)
        ]),
    "Truth/GLM/GLMNet" => NamedTuple{model_keys}(
        [
            sim_model, 
            PopGenEstimatorComparison.treatment_model(all_models, flavor=:GLM),
            PopGenEstimatorComparison.treatment_model(all_models, flavor=:GLMNet)
        ]),
    "Truth/GLMNet/GLM" => NamedTuple{model_keys}(
        [
            sim_model, 
            PopGenEstimatorComparison.treatment_model(all_models, flavor=:GLMNet),
            PopGenEstimatorComparison.treatment_model(all_models, flavor=:GLM)
        ]),
    "Truth/GLMNet/GLMNet" => NamedTuple{model_keys}(
        [
            sim_model, 
            PopGenEstimatorComparison.treatment_model(all_models, flavor=:GLMNet),
            PopGenEstimatorComparison.treatment_model(all_models, flavor=:GLMNet)
        ])
]
Ψ̂s = [name => TMLEE(model_comb) for (name, model_comb) in model_combinations]

Ψ̂₀ = approx_ATE(Ψ, sim_mach, sim_dataset)
sim_results = PopGenEstimatorComparison.run_multiple_estimators(Ψ, sim_dataset, Ψ̂s)
plot_errorbars(sim_results, Ψ̂₀)

#### Problem
η = TMLE.get_relevant_factors(Ψ)
cache = Dict()

models = NamedTuple{(Symbol("A49 Bacterial infection of unspecified site"), :rs4962406, :rs12785878)}(
    [
        PopGenEstimatorComparison.outcome_model(all_models, Ψ.outcome, dataset; flavor=:GLM), 
        PopGenEstimatorComparison.treatment_model(all_models, flavor=:GLM),
        PopGenEstimatorComparison.treatment_model(all_models, flavor=:GLM)
    ])

tmle_pb = TMLEE(models; ps_lowerbound=1e-8)
result_pb, cache = tmle_pb(Ψ, sim_dataset, cache=cache);
result_pb
ĝ1 = cache[η.propensity_score[1]][2]
ĝ2 = cache[η.propensity_score[2]][2]
ĝ = TMLE.likelihood(ĝ1, sim_dataset) .* TMLE.likelihood(ĝ2, sim_dataset)

new_models = NamedTuple{(Symbol("A49 Bacterial infection of unspecified site"), :rs4962406, :rs12785878)}(
    [
        PopGenEstimatorComparison.outcome_model(all_models, Ψ.outcome, dataset; flavor=:GLMNet), 
        LogisticClassifier(lambda=0.001),
        PopGenEstimatorComparison.treatment_model(all_models, flavor=:GLMNet)
    ])

tmle = TMLEE(new_models)
result, cache = tmle(Ψ, sim_dataset, cache=cache);
result

ĝ1new = cache[η.propensity_score[1]][2]
ĝ2new = cache[η.propensity_score[2]][2]

classes = ["AA", "AG", "GG"]
classes = ["TT", "GT", "GG"]

fig = Figure()
ax = Axis(fig[1, 1], xlabel="Unregularized GLM", ylabel="Regularized GLM")
for class_id in 1:3
    g1_pdf = pdf.(predict(ĝ2.machine), classes[class_id])
    g1_new_pdf = pdf.(predict(ĝ2new.machine), classes[class_id])
    scatter!(ax, g1_pdf, g1_new_pdf, label=classes[class_id]) 
end
ablines!(ax, 0, 1, color=:green)
axislegend(ax, position=:lt)
fig

# Fitering extreme propensity score samples
results = []
for threshold in [0., 0.002, 0025, 0.003, 0.004, 0.005]
    indices = findall(x -> x < threshold, ĝ)
    filtered_dataset = sim_dataset[Not(indices), :]
    result_filtered, _ = tmle_pb(Ψ, filtered_dataset,);
    push!(results, string(threshold) => result_filtered)
end
plot_errorbars(results, Ψ̂₀)

### Model evaluation

nfolds = 6
resampling = StratifiedCV(nfolds=nfolds)
ps1 = η.propensity_score[1]
shuffled_sim_dataset = shuffle(sim_dataset)
Xg = shuffled_sim_dataset[!, collect(ps1.parents)]
yg = shuffled_sim_dataset[!, ps1.outcome]
perfs = []
λs = [100, 10, 1, 0.1, 0.001, 0.0001, 0.00001, 1e-5, 1e-6, 1e-7, 0]
for λ in λs
    push!(
        perfs, 
        evaluate(LogisticClassifier(lambda=λ), Xg, yg, measure=log_loss, resampling=resampling)
    )
end
glmnetperf = evaluate(PopGenEstimatorComparison.GLMNetClassifier(), Xg, yg, measure=log_loss, resampling=resampling)

nmodels = length(λs)
fig = Figure()
ax = Axis(fig[1, 1], xticks = (1:nmodels, [string(λs[i]) for i in 1:nmodels]))
losses = [perf.measurement[1] for perf in perfs]
stdes = [1.96*sqrt(var(perf.per_fold[1])/nfolds) for perf in perfs]
errorbars!(ax, 1:nmodels, losses, stdes,
        color = range(0, 1, length = nmodels),
        whiskerwidth = 10)
hlines!(ax, glmnetperf.measurement[1], label="GLMnet")
axislegend(ax)
fig


## Rerun estimation without those samples

PopGenEstimatorComparison.coercetypes!(dataset, Ψ)
# Estimators
model_keys = tuple(Ψ.outcome, keys(Ψ.treatment_values)...)
glm_η̂s = NamedTuple{model_keys}(
    [PopGenEstimatorComparison.outcome_model(all_models, Ψ.outcome, dataset; flavor=:GLM), (PopGenEstimatorComparison.treatment_model(all_models, flavor=:GLM) for _ in 1:length(model_keys) - 1)...]
)
glmnet_η̂s = NamedTuple{model_keys}(
    [PopGenEstimatorComparison.outcome_model(all_models, Ψ.outcome, dataset; flavor=:GLMNet), (PopGenEstimatorComparison.treatment_model(all_models, flavor=:GLMNet) for _ in 1:length(model_keys) - 1)...]
)
sl_η̂s = NamedTuple{model_keys}(
    [PopGenEstimatorComparison.outcome_model(all_models, Ψ.outcome, dataset; flavor=:SL), (PopGenEstimatorComparison.treatment_model(all_models, flavor=:SL) for _ in 1:length(model_keys) - 1)...]
)
Ψ̂s = [
    "TMLE(GLM)" => TMLEE(glm_η̂s, weighted=false, ps_lowerbound=0.005),
    "CV-TMLE(GLM)" => TMLEE(glm_η̂s, weighted=false, resampling=cv, ps_lowerbound=0.005),
    "CV-TMLE(GLMNet)" => TMLEE(glmnet_η̂s, weighted=false, resampling=cv, ps_lowerbound=0.005),
    "TMLE(SL)" => TMLEE(sl_η̂s, weighted=false, ps_lowerbound=0.005),
    "CV-TMLE(SL)" => TMLEE(sl_η̂s, weighted=false, resampling=cv, ps_lowerbound=0.005),
]
results = []
cache = Dict()
for (Ψ̂name, Ψ̂) in Ψ̂s
    result, cache = Ψ̂(Ψ, dataset; cache=cache);
    push!(results, Ψ̂name => result)
end
plot_errorbars(results, Ψ̂₀)
