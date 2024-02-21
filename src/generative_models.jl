### Random Generator

struct RandomDatasetGenerator
    confounders_distribution
    treatments_distribution
    outcome_distribution
end

function RandomDatasetGenerator(;
    confounders_distribution=MvNormal(ones(6), Diagonal(ones(6))),
    treatments_distribution=Bernoulli(0.01),
    outcome_distribution=Bernoulli(0.01)
    )
    return RandomDatasetGenerator(
        confounders_distribution, 
        treatments_distribution, 
        outcome_distribution
    )
end

maybe_transpose(x::AbstractVector) = x
maybe_transpose(x) = transpose(x)

function sample_from_distribution(d::Distribution, n; rng=Random.default_rng(), prefix=:V)
    samples = maybe_transpose(rand(rng, d, n))
    dim = size(samples, 2)
    colnames = Tuple(Symbol(prefix, i) for i in 1:dim)
    return NamedTuple{colnames}([samples[:, i] for i in 1:dim])
end

function sample_from_distribution(ds, n; rng=Random.default_rng(), prefix=:V)
    samples = [sample_from_distribution(d, n; rng=rng, prefix=Symbol(prefix, index)) for (index, d) in enumerate(ds)]
    return merge(samples)
end

function sample(generator::RandomDatasetGenerator, n::Int; rng=Random.default_rng())
    W = PopGenEstimatorComparison.sample_from_distribution(generator.confounders_distribution, n; rng=rng, prefix=:W_)
    T = sample_from_distribution(generator.treatments_distribution, n;rng=rng, prefix=:T_)
    Y = rand(rng, generator.outcome_distribution, n)
    return (Y=Y, T..., W...)
end