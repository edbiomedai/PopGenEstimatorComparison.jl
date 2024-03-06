
struct DensityEstimationSampler
    variables::NamedTuple{(:outcome, :treatments, :confounders, :outcome_extra_covariates)}
    outcome_sampler::NeuralNetworkEstimator
    treatment_samplers::Tuple{Vararg{NeuralNetworkEstimator}}
end

function DensityEstimationSampler(dataset, outcome, treatments; 
    confounders=("PC1", "PC2", "PC3", "PC4", "PC5", "PC6"), 
    outcome_extra_covariates=("Age-Assessment", "Genetic-Sex"),
    hidden_sizes = (20,),
    K = 3,
    verbosity=1,
    kwargs...
    )
    variables = variables_from_args(outcome, treatments, confounders, outcome_extra_covariates)

    # Outcome Sampler
    verbosity > 1 && @info("Training Outcome Sampler.")
    X, y = X_y(
        dataset, 
        outcome_model_inputs(variables), 
        variables.outcome
    )
    outcome_sampler = get_neural_net(X, y, hidden_sizes; K=K, kwargs...)
    train!(outcome_sampler, X, y, verbosity=verbosity-1)

    # Treatment Samplers
    treatment_samplers = []
    for treatment in variables.treatments
        verbosity > 1 && @info(string("Training Treatment Sampler (", treatment, ")."))
        X, y = X_y(dataset, propensity_score_inputs(variables), treatment)
        treatment_sampler = get_neural_net(X, y, hidden_sizes; K=K, kwargs...)
        train!(treatment_sampler, X, y, verbosity=verbosity-1)
        push!(treatment_samplers, treatment_sampler)
    end

    return DensityEstimationSampler(
        variables,
        outcome_sampler,
        Tuple(treatment_samplers)
    )
end

function sample_from(sampler::DensityEstimationSampler, origin_dataset; n=100)
    variables = sampler.variables

    sampled_dataset = sample_from(origin_dataset, confounders_and_covariates(variables); n=n)
    for (treatment_sampler, treatment) in zip(sampler.treatment_samplers, sampler.variables.treatments)
        sampled_dataset[!, treatment] = sample_from(
            treatment_sampler, 
            sampled_dataset[!, PopGenEstimatorComparison.propensity_score_inputs(variables)], 
            PopGenEstimatorComparison.getlabels(origin_dataset[!, treatment])
        ) 
    end

    outcome = sampler.variables.outcome
    sampled_dataset[!, outcome] = sample_from(
        sampler.outcome_sampler, 
        sampled_dataset[!, outcome_model_inputs(variables)], 
        getlabels(origin_dataset[!, outcome])
    )
    return sampled_dataset
end

