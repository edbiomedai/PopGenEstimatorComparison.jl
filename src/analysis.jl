mean_bias(estimates, Ψ₀) = mean(Ψ̂.estimate .- Ψ₀ for Ψ̂ in estimates)

covers(Ψ̂, Ψ₀; alpha=0.05) =
    pvalue(significance_test(Ψ̂, Ψ₀)) > alpha

covers(Ψ̂::TMLE.ComposedEstimate, Ψ₀; alpha=0.05) =
    [covers(Ψ̂ᵢ,Ψ₀ᵢ; alpha=alpha) for (Ψ̂ᵢ, Ψ₀ᵢ) in zip(Ψ̂.estimates, Ψ₀)]

mean_coverage(estimates, Ψ₀) = mean(covers(Ψ̂, Ψ₀) for Ψ̂ in estimates)

function analysis(
    estimation_results_file, 
    dataset_file;
    density_estimates_prefix=nothing
    )
    estimation_results = jldopen(io -> io["results"], estimation_results_file)
    origin_dataset = TargetedEstimation.instantiate_dataset(dataset_file)
    other_columns = ["REPEAT_ID", "SAMPLE_SIZE", "RNG_SEED"]
    estimates_columns = filter(∉(other_columns), names(estimation_results))
    estimation_results.ESTIMAND = [x.estimand for x in estimation_results[!, first(estimates_columns)]]
    summary_statistics = []
    for (key, group) in pairs(groupby(results, [:ESTIMAND, :SAMPLE_SIZE]))
        Ψ = key.ESTIMAND
        sample_size = key.SAMPLE_SIZE
        sampler = PopGenEstimatorComparison.get_sampler(density_estimates_prefix, [Ψ])
        Ψ₀ = PopGenEstimatorComparison.true_effect(Ψ, sampler, origin_dataset; n=500_000)
        for estimator in estimates_columns
            estimates = filter(x -> !(x isa TargetedEstimation.FailedEstimate), group[!, estimator])
            n_repeats = length(estimates)
            bias = mean_bias(estimates, Ψ₀)
            coverage = mean_coverage(estimates, Ψ₀)
            push!(
                summary_statistics, 
                [Ψ, estimator, sample_size, n_repeats, bias, coverage]
            )
        end
    end
    DataFrame(summary_statistics, [:ESTIMAND, :ESTIMATOR, :SAMPLE_SIZE, :N_REPEATS, :BIAS, :COVERAGE])
end