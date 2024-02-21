using CSV
using DataFrames
using CairoMakie
using PopGenEstimatorComparison
using Arrow

function make_frequency_plot(white_individuals, all_individuals, Ψ)
    treatments = collect(keys(Ψ.treatment_values))
    all_columns = unique(Iterators.flatten(
        [[Ψ.outcome], 
        keys(Ψ.treatment_values), 
        values(Ψ.treatment_confounders)...,
        Ψ.outcome_extra_covariates]
    ))
    white_individuals = dropmissing(white_individuals, all_columns)
    all_individuals = dropmissing(all_individuals, all_columns)
    # Non white individuals
    non_white_ids = DataFrame(SAMPLE_ID=setdiff(all_individuals.SAMPLE_ID, white_individuals.SAMPLE_ID))
    non_whites = innerjoin(
        all_individuals[!, [:SAMPLE_ID, Ψ.outcome, keys(Ψ.treatment_values)...]],
        non_white_ids,
        on=:SAMPLE_ID
    )

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
        # Non Whites
        treatment_non_whites = DataFrames.combine(
            groupby(non_whites, treatment), 
            nrow
            )
        treatment_non_whites.freq = treatment_non_whites.nrow ./ size(non_whites, 1)
        treatment_non_whites.group .= 3

        # Combine for plot
        stats = vcat(treatment_white, treatment_all, treatment_non_whites)
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
    # Non Whites
    ncases_non_whites = DataFrames.combine(
        groupby(non_whites, treatments), 
        Ψ.outcome => sum => :ncases
    )
    ncases_non_whites.freq = ncases_non_whites.ncases ./ size(non_whites, 1)
    ncases_non_whites.group .= 3

    # Combine for plot
    stats = vcat(ncases_white, ncases_all, ncases_non_whites)
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

    # Shared Legend and title
    elements = [PolyElement(polycolor = colors[i]) for i in 1:3]
    elements = vcat(elements, dots)
    Legend(fig[:, 3], elements, ["White", "All", "Non Whites", "Contribute to Ψ"])
    
    treatment_spec = join(
        (string(treatment, ": ", treatment_values.control, "⟶", treatment_values.case) 
        for (treatment, treatment_values) ∈ zip(keys(Ψ.treatment_values), Ψ.treatment_values)), 
        ", "
    )
    title = string(replace(string(typeof(Ψ)), "TMLE.Statistical" => ""), ": ", Ψ.outcome, ", " , treatment_spec)
    Label(fig[0, :], title, fontsize = 14)
    return fig
end

datasets = Dict(
    "All" => "data/all_population_data.arrow",
    "White" => "data/white_population_data.arrow",
)
estimands_file = "data/problematic_estimands.csv"

# Load Data
problematic_estimands = CSV.read(estimands_file, DataFrame)
all_individuals = Arrow.Table(datasets["All"]) |> DataFrame
white_individuals = Arrow.Table(datasets["White"]) |> DataFrame

# Estimand
id = 6
row = problematic_estimands[id, :]
Ψ = PopGenEstimatorComparison.estimand_from_results_row(row)

make_frequency_plot(white_individuals, all_individuals, Ψ)
