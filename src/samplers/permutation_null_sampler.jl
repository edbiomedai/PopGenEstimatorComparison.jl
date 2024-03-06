"""
The Permutation-Null-Sampler keeps the marginal distributions of each variable in the original dataset
intact while disrupting the causal relationships between them. This is done by:
    1. Sampling from (W, C)
    2. Permuting each T
    3. Permuting Y
"""
struct PermutationNullSampler
    variables::NamedTuple{(:outcome, :treatments, :confounders, :outcome_extra_covariates)}
end

function PermutationNullSampler(outcome, treatments; 
    confounders=("PC1", "PC2", "PC3", "PC4", "PC5", "PC6"), 
    outcome_extra_covariates=("Age-Assessment", "Genetic-Sex")
    )
    variables = variables_from_args(outcome, treatments, confounders, outcome_extra_covariates)
    return PermutationNullSampler(variables)
end


function sample_from(sampler::PermutationNullSampler, origin_dataset; n=100)
    variables = sampler.variables
    sampled_dataset = sample_from(origin_dataset, confounders_and_covariates(variables); n=n)
    for treatment in variables.treatments
        sampled_dataset[!, treatment] = StatsBase.sample(origin_dataset[!, treatment], n, replace=true)
    end
    sampled_dataset[!, variables.outcome] = StatsBase.sample(origin_dataset[!, variables.outcome], n, replace=true)
    return sampled_dataset
end