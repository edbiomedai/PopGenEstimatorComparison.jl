using Makie
using CairoMakie
using DataFrames
using PopGenEstimatorComparison
using JLD2
using TargetedEstimation
using HypothesisTests
using Serialization
using TMLE

"""
This functions assumes that the losses contained in the density estimates files are indexed by:
1 → Sieve Neural Network
2 → GLM (Baseline)
"""
function get_density_lossratio_info(density_estimates_prefix)
    lossratio_table = DataFrame()
    outcome_mean_id = 1
    propensity_score_id = 1
    for (distribution_id, file) in enumerate(PopGenEstimatorComparison.files_matching_prefix(density_estimates_prefix))
        jldopen(file) do io
            outcome = string(io["outcome"])
            parents = join(io["parents"], ",")
            losses = io["metrics"]
            test_rel_diff = -100(losses[1].test_loss - losses[2].test_loss) / losses[2].test_loss
            train_rel_diff = -100(losses[1].train_loss - losses[2].train_loss) / losses[2].train_loss
            if occursin(r"^rs[0-9]*", outcome)
                distribution = outcome
                type = "Propensity Score"
                distribution_id = propensity_score_id
                propensity_score_id += 1
            else
                distribution = string(
                    first(outcome, 10), "... | ", 
                    join((p for p in string.(io["parents"]) if occursin(r"^rs[0-9]*", p)), ", ")
                )
                type = "Outcome Mean"
                distribution_id = outcome_mean_id
                outcome_mean_id += 1
            end
            row = (
                DISTRIBUTION_ID = distribution_id, 
                DISTRIBUTION = distribution,
                OUTCOME = outcome,
                PARENTS = parents,
                TEST_REL_DIFF = test_rel_diff,
                TRAIN_REL_DIFF = train_rel_diff,
                TYPE = type,
                TEST_COLOR = test_rel_diff > 0 ? :blue : :orange,
                TRAIN_COLOR = train_rel_diff > 0 ? :blue : :orange
                )
            push!(lossratio_table, row)
        end
    end
    return lossratio_table
end

function density_estimates_barplot!(loss_table; set=:TEST, title="Outcome Means (Test Set)")
    loss_column = Symbol(set, :_REL_DIFF)
    loss_color = Symbol(set, :_COLOR)
    sort!(loss_table, loss_column)
    loss_table.DISTRIBUTION_ID = 1:nrow(loss_table)

    distributions_labels = loss_table.DISTRIBUTION
    yticks = (1:length(distributions_labels), distributions_labels)
    
    fig = Figure(size=(1000, 800))
    ax = Axis(fig[1, 1], title=title, ylabel="Density", xlabel="LogLoss Relative Improvement (%)", yticks=yticks)
    barplot!(ax,
        loss_table.DISTRIBUTION_ID, 
        loss_table[!, loss_column],
        color=loss_table[!, loss_color],
        direction=:x,
        )
    vlines!(ax, 0, color=:black)
    hlines!(ax, findfirst(>(0), loss_table[!, loss_column]), color=:black, linestyle=:dash)
    return fig
end

function density_estimates_barplots(outdir, density_estimates_prefix)
    loss_table = get_density_lossratio_info(density_estimates_prefix)
    # Outcome Means
    outcome_means = filter(x -> x.TYPE == "Outcome Mean", loss_table)
    fig = density_estimates_barplot!(outcome_means; set=:TEST, title="Outcome Means (Test Set)")
    save(joinpath(outdir, "outcome_means_de_test_performance.png"), fig)
    fig = density_estimates_barplot!(outcome_means; set=:TRAIN, title="Outcome Means (Train Set)")
    save(joinpath(outdir, "outcome_means_de_train_performance.png"), fig)
    # Propensity Scores
    propensity_scores = filter(x -> x.TYPE == "Propensity Score", loss_table)
    fig = density_estimates_barplot!(propensity_scores; set=:TEST, title="Propensity Scores (Test Set)")
    save(joinpath(outdir, "propensity_scores_de_test_performance.png"), fig)
    fig = density_estimates_barplot!(propensity_scores; set=:TRAIN, title="Propensity Scores (Train Set)")
    save(joinpath(outdir, "propensity_scores_de_train_performance.png"), fig)
end

function KSTest_plots(dataset, density_estimates_prefix, outdir)
    for file in PopGenEstimatorComparison.files_matching_prefix(density_estimates_prefix)
        jldopen(file) do io
            outcome = io["outcome"]
            parents = io["parents"]
            variables = [outcome, parents...]
            nomissing_dataset = dropmissing(DataFrames.select(dataset, variables), variables)
            TargetedEstimation.coerce_types!(nomissing_dataset, [outcome, parents...])
            X = DataFrames.select(nomissing_dataset, parents)
            y = nomissing_dataset[!, outcome]
            density_estimate = io["sieve-neural-net"]
            y_sampled = sample_from(
                density_estimate.neural_net_estimator.model, 
                PopGenEstimatorComparison.transpose_table(density_estimate.neural_net_estimator, X),
                density_estimate.neural_net_estimator.labels
            )
        end
    end
end

function get_genotypes_stats(dataset, variant)
    stats = sort(DataFrames.combine(groupby(dataset, variant, skipmissing=true), nrow => :FREQ), :FREQ)
    n = sum(stats.FREQ)
    maf = if size(stats, 1) == 3
        (stats.FREQ[1] + stats.FREQ[2]) / 2n 
    else
        throw(ArgumentError(string("Can't compute MAF for variant: ", variant)))
    end
    stats.FREQ = stats.FREQ/n
    rename!(stats, variant => :GENOTYPES)
    stats.VARIANT .= variant
    stats.MAF .= maf
    stats.ALLELE_ID = [1, 2, 3]

    return stats
end

function plot_dataset(dataset, outdir)
    descriptive_dir = joinpath(outdir, "descriptive")
    isdir(descriptive_dir) || mkdir(descriptive_dir)
    variables = PopGenEstimatorComparison.variables_from_dataset(dataset)
    # Variants
    variants_stats = DataFrame()
    for (variant_id, variant) in enumerate(variables.variants)
        variant_stats = get_genotypes_stats(dataset, variant)
        variant_stats.VARIANT_ID .= variant_id
        variants_stats = vcat(variants_stats, variant_stats)
    end

    colors = Makie.wong_colors()
    fig = Figure(size=(1000, 700))
    ax = Axis(fig[1, 1], 
        title="Variants's Genotypes Frequencies", 
        xticks=(1:length(variables.variants), variables.variants))
    barplot!(ax, 
        variants_stats.VARIANT_ID, 
        variants_stats.FREQ,
        color=colors[variants_stats.ALLELE_ID],
        dodge = variants_stats.ALLELE_ID,
        bar_labels=variants_stats.GENOTYPES, 
        color_over_bar=:white,
    )
    save(joinpath(descriptive_dir, "variants.png"), fig)
    # Traits
    binary_traits = []
    continuous_traits = []
    for trait ∈ variables.outcomes
        if eltype(dataset[!, trait]) <: Union{Bool, Missing}
            push!(binary_traits, trait)
        else
            push!(continuous_traits, trait)
        end
    end
    ncases = [sum(skipmissing(dataset[!, trait])) for trait in binary_traits]
    ## Binary Traits
    fig = Figure()
    ax = Axis(fig[1, 1], 
        title="Binary Traits",
        xscale=log10,
        yticks=(1:length(ncases), binary_traits)
    )
    barplot!(ax, 
        1:length(ncases), 
        ncases, 
        bar_labels=:y,
        flip_labels_at=15000,
        label_formatter = x-> "$(Int(x))",
        direction=:x
    )
    save(joinpath(descriptive_dir, "binary_traits.png"), fig)
    ## Continuous Traits
    for trait in continuous_traits
        fig = Figure()
        ax = Axis(fig[1, 1], title=trait)
        hist!(ax, collect(skipmissing(dataset[!, trait])), bins=100)
        save(joinpath(descriptive_dir, string(trait, ".png")), fig)
    end
end

function main(;
    outdir="outputs",
    estimands_file = joinpath("assets", "estimands", "estimands.jls"),
    dataset_file=joinpath("dataset", "results", "dataset.arrow"),
    density_estimates_prefix=joinpath("simulation", "results", "density_estimation", "density_estimates/de")
    )
    isdir(outdir) || mkdir(outdir)
    dataset = TargetedEstimation.instantiate_dataset(dataset_file)
    # Descriptive Plots
    plot_dataset(dataset, outdir)
    # Density Estimates analysis
    density_estimates_barplots(outdir, density_estimates_prefix)
    # KSTest_plots(dataset, density_estimates_prefix, outdir)
end

main()