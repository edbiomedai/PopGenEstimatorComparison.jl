using CairoMakie
using DataFrames
using PopGenEstimatorComparison
using JLD2

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

function main(;
    outdir="outputs", 
    density_estimates_prefix="simulation/results/density_estimation/density_estimates/de"
    )
    isdir(outdir) || mkdir(outdir)
    density_estimates_barplots(outdir, density_estimates_prefix)
end

main()