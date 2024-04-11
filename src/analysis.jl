bias(Ψ̂::T, Ψ₀::T) where T = Ψ̂ .- Ψ₀

bootstrap_bias(estimates, Ψ₀) = mean(bias(Ψ̂.estimate, Ψ₀) for Ψ̂ in estimates)

bootstrap_variance(estimates) = var([Ψ̂.estimate for Ψ̂ ∈ estimates])

function ci_width(Ψ̂, Ψ₀)
    lb, ub = confint(significance_test(Ψ̂, Ψ₀))
    return abs(ub-lb)
end

bootstrap_ci_width(estimates, Ψ₀) = mean(ci_width(Ψ̂, Ψ₀) for Ψ̂ ∈ estimates)

covers(Ψ̂, Ψ₀; alpha=0.05) =
    pvalue(significance_test(Ψ̂, Ψ₀)) > alpha

bootstrap_coverage(estimates, Ψ₀) = mean(covers(Ψ̂, Ψ₀) for Ψ̂ in estimates)

function collect_summary_statistics(results, true_effects, estimators, sample_size)
    results_subset = results[estimators][sample_size]
    estimands = Any[]
    estimator_names = Symbol[]
    sample_sizes = Int[]
    biases = Float64[]
    variances = Float64[]
    mses = Float64[]
    coverages = Float64[]
    ci_widths = Float64[]
    for (key, group) in pairs(groupby(results_subset, :ESTIMAND))
        Ψ = key.ESTIMAND
        Ψ₀ = true_effects[Ψ]
        for estimator in estimators
            estimates = group[!, estimator]
            b = bootstrap_bias(estimates, Ψ₀)
            v = bootstrap_variance(estimates)
            mse = b^2 + v
            c = bootstrap_coverage(estimates, Ψ₀)
            w = bootstrap_ci_width(estimates, Ψ₀)
            push!(estimands, Ψ)
            push!(estimator_names, estimator)
            push!(sample_sizes, sample_size)
            push!(biases, b)
            push!(variances, v)
            push!(mses, mse)
            push!(coverages, c)
            push!(ci_widths, w)
        end
    end
    return DataFrame(
        ESTIMAND=estimands,
        ESTIMATOR=estimator_names,
        SAMPLE_SIZE=sample_sizes,
        BIAS=biases,
        VARIANCE=variances,
        MSE=mses,
        COVERAGE=coverages,
        CI_WIDTH=ci_widths
    )
end

function collect_summary_statistics(results, true_effects)
    summary_statistics = []
    for (estimators, estimators_result) ∈ results
        for sample_size ∈ keys(estimators_result)
            push!(
                summary_statistics,
                collect_summary_statistics(results, true_effects, estimators, sample_size)
            )
        end
    end
    return reduce(vcat, summary_statistics)
end

function get_true_effect_sizes(estimands_prefix, density_estimates_prefix, origin_dataset=nothing; n=500_000)
    estimands = reduce(
        vcat, 
        deserialize(f).estimands for f ∈ files_matching_prefix(estimands_prefix)
    )

    true_effects = map(estimands) do Ψ
        sampler = PopGenEstimatorComparison.get_sampler(density_estimates_prefix, [Ψ])
        true_effect(Ψ, sampler, origin_dataset; n=n)
    end
    return Dict(zip(estimands, true_effects))
end

function update_results_1d!(results, indices, Ψ̂::TMLE.ComposedEstimate, index)
    append!(results, Ψ̂.estimates)
    append!(indices, [index for _ in 1:length(Ψ̂.estimates)])
end

function update_results_1d!(results, indices, Ψ̂::TMLE.Estimate, index)
    push!(results, Ψ̂)
    push!(indices, index)
end

update_results_1d!(results, indices, Ψ̂::TargetedEstimation.FailedEstimate, index) = nothing

function unpack_results1D(results, estimators, sample_size)
    results_subset = results[estimators][sample_size]
    results_subset.INDEX = 1:nrow(results_subset)

    results1D = DataFrames.select(results_subset, [:INDEX, :REPEAT_ID, :RNG_SEED])
    for estimator in estimators
        estimator_results1D = []
        indices = Int64[]
        for (index, Ψ̂) in zip(results_subset.INDEX, results_subset[!, estimator])
            update_results_1d!(estimator_results1D, indices, Ψ̂, index)
        end
        estimator_results1D = DataFrame([estimator_results1D, indices], [estimator, :INDEX])
        results1D = innerjoin(results1D, estimator_results1D; on=:INDEX)
    end
    results1D[!, :ESTIMAND] = [x.estimand for x in results1D[!, first(estimators)]]
    return select!(results1D, Not(:INDEX))
end

function unpack_results1D(results)
    results1D = Dict()
    for (estimators, estimators_result) ∈ results
        for sample_size ∈ keys(estimators_result)
            results1D_subset = unpack_results1D(results, estimators, sample_size)
            if haskey(results1D, estimators)
                results1D[estimators][sample_size] = results1D_subset
            else
                results1D[estimators] = Dict(sample_size => results1D_subset)
            end
        end
    end
    return results1D
end

update_true_effects1D!(true_effects1D, Ψ, effect) =
    true_effects1D[Ψ] = effect


function update_true_effects1D!(true_effects1D, Ψ::ComposedEstimand, effects::AbstractVector)
    for (Ψᵢ, effect) ∈ zip(Ψ.args, effects)
        update_true_effects1D!(true_effects1D, Ψᵢ, effect)
    end
end

function unpack_true_effects1D(true_effects)
    true_effects1D = Dict()
    for (Ψ, effect) in true_effects
        update_true_effects1D!(true_effects1D, Ψ, effect)
    end
    return true_effects1D
end

function estimands_by_sample_size_plots(summary_statistics)
    estimator_ids = sort(unique(summary_statistics.ESTIMATOR))
    estimator_ids = DataFrame(ESTIMATOR=estimator_ids, ESTIMATOR_ID=1:length(estimator_ids))
    yticks = (estimator_ids.ESTIMATOR_ID, string.(estimator_ids.ESTIMATOR))
    summary_statistics = innerjoin(summary_statistics, estimator_ids, on=:ESTIMATOR)
    (key, group) = first(pairs(groupby(summary_statistics, [:ESTIMAND, :SAMPLE_SIZE])))
    for (key, group) in pairs(groupby(summary_statistics, [:ESTIMAND]))
        max_sample_size = maximum(group.SAMPLE_SIZE)
        fig = Figure(size = (1000, 700))
        # Top Layout
        top = fig[1, 1:3] = GridLayout()
        # Bias
        ax = Axis(top[1, 1], xlabel="Bias", yticks=yticks)
        scatter!(ax, group.BIAS, group.ESTIMATOR_ID, markersize=20group.SAMPLE_SIZE./max_sample_size)
        vlines!(ax, 0, color=:black)
        # Variance
        ax = Axis(top[1, 2], xlabel="Variance", yticks=yticks)
        scatter!(ax, group.VARIANCE, group.ESTIMATOR_ID, markersize=20group.SAMPLE_SIZE./max_sample_size)
        vlines!(ax, 0, color=:black)
        # MSE
        ax = Axis(top[1, 3], xlabel="MSE", yticks=yticks)
        scatter!(ax, group.MSE, group.ESTIMATOR_ID, markersize=20group.SAMPLE_SIZE./max_sample_size)
        vlines!(ax, 0, color=:black)
        # Bottom Layout
        bottom = fig[2, 1:3] = GridLayout()
        ## CI Width
        ax = Axis(bottom[1, 1], xlabel="CI Width", yticks=yticks, xtickformat = "{:.2f}")
        scatter!(ax, group.CI_WIDTH, group.ESTIMATOR_ID, markersize=20group.SAMPLE_SIZE./max_sample_size)
        ## Coverage
        ax = Axis(bottom[1, 2], xlabel="Coverage", yticks=yticks)
        scatter!(ax, group.COVERAGE, group.ESTIMATOR_ID, markersize=10group.SAMPLE_SIZE./max_sample_size)
        vlines!(ax, 0.95, color=:black)
        # Title
        Ψ = key.ESTIMAND
        estimand_type = replace(string(typeof(Ψ)), "TMLE.Statistical" => "")
        treatments = join((string(tn, ": ", cc.control, " → ", cc.case) for (tn, cc) in zip(keys(Ψ.treatment_values), Ψ.treatment_values)), ", ")
        title = string(estimand_type, ": ", Ψ.outcome, ", ", treatments)
        Label(top[1, 1:3, Top()], title, valign = :bottom, font = :bold, padding = (0, 0, 5, 0))
        display(fig)
    end
end

function analysis1D(results, true_effects, out_dir_1D)
    results1D = PopGenEstimatorComparison.unpack_results1D(results)
    true_effects1D = unpack_true_effects1D(true_effects)
    summary_statistics = collect_summary_statistics(results1D, true_effects1D)
    jldsave(joinpath(out_dir_1D, "summary_stats.hdf5"), results=summary_statistics)
end

function analyse(
    results_file,
    estimands_prefix;
    out_dir="analysis_results",
    n=500_000,
    dataset_file=nothing,
    density_estimates_prefix=nothing
    )
    isdir(out_dir) || mkdir(out_dir)
    origin_dataset = dataset_file !== nothing ? TargetedEstimation.instantiate_dataset(dataset_file) : nothing
    # Analysis of 1-dimensional effects
    out_dir_1D = joinpath(out_dir, "analysis1D")
    isdir(out_dir_1D) || mkdir(out_dir_1D)
    results = jldopen(io -> io["results"], results_file)
    true_effects = PopGenEstimatorComparison.get_true_effect_sizes(estimands_prefix, density_estimates_prefix, origin_dataset; n=n)
    PopGenEstimatorComparison.analysis1D(results, true_effects, out_dir_1D)
    # Analysis of multi-dimensional effects
end