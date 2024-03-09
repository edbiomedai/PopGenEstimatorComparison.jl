"""
The Permutation-Null-Sampler keeps the marginal distributions of each variable in the original dataset
intact while disrupting the causal relationships between them. This is done by:
    1. Sampling from (W, C)
    2. Permuting each T
    3. Permuting Y
"""
struct PermutationNullSampler
    confounders_and_covariates
    other_variables
    function PermutationNullSampler(estimands)
        # Check confounders and covariates are the same for all estimands
        confounders_and_covariates = confounders_and_covariates_set(first(estimands))
        other_variables = Set{Symbol}([])
        for Ψ in estimands
            @assert confounders_and_covariates_set(Ψ) == confounders_and_covariates "All estimands should share the same confounders and covariates."
            push!(other_variables, get_outcome(Ψ))
            push!(other_variables, get_treatments(Ψ)...)
        end
        return new(confounders_and_covariates, other_variables)
    end
end

function PermutationNullSampler(outcome, treatments; 
    confounders=("PC1", "PC2", "PC3", "PC4", "PC5", "PC6"), 
    outcome_extra_covariates=("Age-Assessment", "Genetic-Sex")
    )
    variables = variables_from_args(outcome, treatments, confounders, outcome_extra_covariates)
    return PermutationNullSampler(variables)
end

function sample_from(sampler::PermutationNullSampler, origin_dataset; n=100)
    sampled_dataset = sample_from(origin_dataset, collect(sampler.confounders_and_covariates); n=n)
    for variable in sampler.other_variables
        sampled_dataset[!, variable] = StatsBase.sample(origin_dataset[!, variable], n, replace=true)
    end
    return sampled_dataset
end