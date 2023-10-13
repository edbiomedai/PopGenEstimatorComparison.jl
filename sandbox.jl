using CSV
using DataFrames
using CairoMakie
using PopGenEstimatorComparison
using Arrow
using TMLE
using MLJBase
import MLJGLMInterface as MLJGLM

function groupby_freqs(data, cols)
    counts = DataFrames.combine(groupby(data, cols, skipmissing=true), nrow)
    counts.freq = counts.nrow ./ sum(counts.nrow)
    return counts
end

function unconfounded_estimate(Ψ::TMLE.StatisticalATE, data)
    treatments = vcat(keys(Ψ.treatment_values)...)
    cols = vcat(Ψ.outcome, treatments)
    nomissing_data = dropmissing(data, cols)
    ncases_per_genotype = DataFrames.combine(groupby(nomissing_data, treatments), Ψ.outcome => sum => :ncases)
    ncases_per_genotype.freq = ncases_per_genotype.ncases ./ size(nomissing_data, 1)
    case = DataFrame(;((key, Ψ.treatment_values[key].case) for key in keys(Ψ.treatment_values))...)
    case = only(innerjoin(ncases_per_genotype, case, on=treatments).freq)
    control = DataFrame(;((key, Ψ.treatment_values[key].control) for key in keys(Ψ.treatment_values))...)
    control = only(innerjoin(ncases_per_genotype, control, on=treatments).freq)
    return case - control
end


datasets = Dict(
    "All" => "data/all_population_data.arrow",
    "White" => "data/white_population_data.arrow",
)
estimands_file = "data/problematic_estimands.csv"
verbosity = 1
id = 6

# Load Data
problematic_estimands = CSV.read(estimands_file, DataFrame)
all_individuals = Arrow.Table(datasets["All"]) |> DataFrame
white_individuals = Arrow.Table(datasets["White"]) |> DataFrame

# Estimand
row = problematic_estimands[id, :]
Ψ = PopGenEstimatorComparison.estimand_from_results_row(row)
y_features = vcat(
    Ψ.outcome_extra_covariates..., 
    first(values(Ψ.treatment_confounders))..., 
    keys(Ψ.treatment_values)...
)

function make_frequncy_plot(white_individuals, all_individuals, Ψ)
    treatments = collect(keys(Ψ.treatment_values))
    all_columns = unique(Iterators.flatten(
        [[Ψ.outcome], 
        keys(Ψ.treatment_values), 
        values(Ψ.treatment_confounders)...,
        Ψ.outcome_extra_covariates]
    ))
    white_individuals = dropmissing(white_individuals, all_columns)
    all_individuals = dropmissing(all_individuals, all_columns)

    colors = Makie.wong_colors()
    fig = Figure()
    # Add treatments frequency plots
    for (treatment_index, treatment) in enumerate(treatments)
        # White
        treatment_white = DataFrames.combine(
            groupby(white_individuals, treatment), 
            nrow
            )
        treatment_white.freq = treatment_white.nrow ./ size(white_individuals, 1)
        treatment_white.group .= 1
        xs = Dict(val => index for (index, val) in enumerate(treatment_white[!, treatment]))
        # All
        treatment_all = DataFrames.combine(
            groupby(all_individuals, treatment), 
            nrow
            )
        treatment_all.freq = treatment_all.nrow ./ size(all_individuals, 1)
        treatment_all.group .= 2
        
        stats = vcat(treatment_white, treatment_all)
        stats.x = [xs[val] for val in stats[!, treatment]]

        ax = Axis(fig[1, treatment_index], title=string(treatment), xticks=(1:length(xs), treatment_white[!, treatment]))
        barplot!(ax, stats.x, stats.freq, dodge=stats.group, color=colors[stats.group])
    end

    # Add outcome frequency plot
    # Whites
    ncases_white = DataFrames.combine(
        groupby(white_individuals, treatments), 
        Ψ.outcome => sum => :ncases
        )
    ncases_white.freq = ncases_white.ncases ./ size(white_individuals, 1)
    ncases_white.group .= 1
    xs = Dict(values(row) => index for (index, row) in enumerate(eachrow(ncases_white[!, treatments])))
    # All
    ncases_all = DataFrames.combine(
        groupby(all_individuals, treatments), 
        Ψ.outcome => sum => :ncases
    )
    ncases_all.freq = ncases_all.ncases ./ size(all_individuals, 1)
    ncases_all.group .= 2
    stats = vcat(ncases_white, ncases_all)
    stats.x = [xs[values(row)] for row in  eachrow(stats[!, treatments])]

    tomark = []
    for row in eachrow(stats)
        if all((row[treatment] ∈ Ψ.treatment_values[treatment]) for treatment in treatments)
            push!(tomark, row.x)
        end
    end
    unique!(tomark)

    ax = Axis(fig[2, :], title=string(Ψ.outcome), xticks = (1:length(xs), [join(row, "/") for row in eachrow(ncases_white[!, treatments])]))
    barplot!(ax, stats.x, stats.freq, dodge=stats.group, color=colors[stats.group])
    dots = scatter!(ax, float.(tomark), zeros(size(tomark, 1)), color=:green, marker=:star8, markersize=30)

    # Shared Legend
    elements = [PolyElement(polycolor = colors[i]) for i in 1:2]
    elements = vcat(elements, dots)
    Legend(fig[:, 3], elements, ["White", "All", "Contribute to Ψ"])
    fig

end
##### Next

groupby_freqs(white_individuals, ["rs4962406", "rs12785878", "A49 Bacterial infection of unspecified site"])

unconfounded_estimate(Ψ, all_individuals)
unconfounded_estimate(Ψ, white_individuals)

y_distr = TMLE.ConditionalDistribution(Ψ.outcome, y_features)
model_keys = tuple(Ψ.outcome, keys(Ψ.treatment_values)...)
glm_η̂s = NamedTuple{model_keys}(
    [
        PopGenEstimatorComparison.outcome_model(all_models, Ψ.outcome, white_individuals; flavor=:GLM), 
        (PopGenEstimatorComparison.treatment_model(all_models, flavor=:GLM) for _ in 1:length(model_keys) - 1)...
    ]
)
tmle = TMLEE(glm_η̂s, weighted=false,)

PopGenEstimatorComparison.coercetypes!(white_individuals, Ψ)
cache_white = Dict()
result_white, cache_white = tmle(Ψ, white_individuals, cache=cache_white, verbosity=verbosity);
result_white
Q_white = cache_white[y_distr][2]
fitted_params(Q_white.machine).logistic_classifier.coefs

cache_all = Dict()
PopGenEstimatorComparison.coercetypes!(all_individuals, Ψ)
result_all, cache_all = tmle(Ψ, all_individuals, cache=cache_all, verbosity=verbosity);
result_all
Q_all = cache_all[y_distr][2]
fitted_params(Q_all.machine).logistic_classifier.coefs