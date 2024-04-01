
best_density_estimator(file::AbstractString) = jldopen(io -> restore!(io["best-estimator"]), file)

struct DensityEstimateSampler
    sources::Vector
    treatment_density_mapping::Dict
    outcome_density_mapping::Dict
end

function DensityEstimateSampler(prefix, estimands)
    densities_dir, _prefix = splitdir(prefix)
    _densities_dir = densities_dir == "" ? "." : densities_dir
    # Create density to file map (There could be more files than actually required)
    density_mapping = Dict()
    for f in readdir(_densities_dir)
        if startswith(f, _prefix)
            jldopen(joinpath(densities_dir, f)) do io
                density_mapping[(Symbol(io["outcome"]) => Tuple(Symbol.(io["parents"])))] = joinpath(densities_dir, f)
            end
        end
    end
    # Create required density to file map
    required_densities = Dict()
    all_parents = Set([])
    all_outcomes = Set([])
    for Ψ in estimands
        for η in TMLE.nuisance_functions_iterator(Ψ)
            required_densities[η.outcome => η.parents] = density_mapping[η.outcome => η.parents]
            union!(all_parents, η.parents)
            push!(all_outcomes, η.outcome)
        end
    end
    # A source is never an outcome
    sources = [x for x in all_parents if x ∉ all_outcomes]
    # A treatment node is both an outcome and a parent
    treatment_density_mapping = Dict(cd => f for (cd, f) in required_densities if (cd[1] ∈ all_parents && cd[1] ∈ all_outcomes))
    # An outcome node is never a parent
    outcome_density_mapping = Dict(cd => f for (cd, f) in required_densities if cd[1] ∉ all_parents)
    return DensityEstimateSampler(sources, treatment_density_mapping, outcome_density_mapping)
end

function sample_from(sampler::DensityEstimateSampler, origin_dataset; n=100)

    sampled_dataset = sample_from(origin_dataset, sampler.sources; n=n)

    for density_mapping in (sampler.treatment_density_mapping, sampler.outcome_density_mapping)
        for ((outcome, parents), file) in density_mapping
            conditional_density_estimate = PopGenEstimatorComparison.best_density_estimator(file)
            sampled_dataset[!, outcome] = sample_from(
                conditional_density_estimate, 
                sampled_dataset[!, collect(parents)], 
                PopGenEstimatorComparison.getlabels(origin_dataset[!, outcome])
            ) 
        end
    end

    return sampled_dataset
end

