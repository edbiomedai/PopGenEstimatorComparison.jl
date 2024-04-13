using TMLE
using Arrow
using DataFrames
using Serialization
using PopGenEstimatorComparison

function getTraits()
    return Symbol.([
        "G35 Multiple sclerosis", 
        "Vitamin D Level",
        "White blood cell (leukocyte) count", 
        "sarcoidosis", 
        "D86 Sarcoidosis", 
        "G35 Multiple sclerosis", 
        "K90-K93 Other diseases of the digestive system",
        "H00-H06 Disorders of eyelid, lacrimal system and orbit", 
        "Trunk fat percentage",
        "Red-Hair"
    ])
end

function getATEs(dataset, confounders, outcome_extra_covariates; traits = getTraits(), positivity_constraint=0.)
    variants = [(:rs117913124,), (:rs2076530,), (:rs12785878,)]
    return reduce(vcat, factorialEstimands(
        ATE, dataset, variant, traits; 
        confounders=confounders, 
        outcome_extra_covariates=outcome_extra_covariates,
        positivity_constraint=positivity_constraint) 
        for variant in variants
    )
end

function getIATEs(dataset, confounders, outcome_extra_covariates; traits = getTraits(), positivity_constraint=0.)
    variants_pairs = [(:rs1805005, :rs6059655), (:rs1805007, :rs6088372)]
    return reduce(vcat, factorialEstimands(
            IATE, dataset, variants_pair, traits;
            confounders=confounders, 
            outcome_extra_covariates=outcome_extra_covariates,
            positivity_constraint=positivity_constraint
        )
        for variants_pair in variants_pairs
    ) 
end

function main()
    DATASET_FILE = joinpath("dataset", "results", "dataset.arrow")
    DESTINATION_DIR = joinpath("assets", "estimands")
    positivity_constraint = 0.
    @assert isdir(DESTINATION_DIR)
    dataset = Arrow.Table(DATASET_FILE) |> DataFrame

    variables = PopGenEstimatorComparison.variables_from_dataset(dataset)

    ATEs = getATEs(
        dataset,
        variables.confounders, 
        variables.outcome_extra_covariates; 
        positivity_constraint=positivity_constraint
    )
    IATEs = getIATEs(
        dataset,
        variables.confounders, 
        variables.outcome_extra_covariates; 
        positivity_constraint=positivity_constraint
    )
    estimands = groups_ordering(vcat(ATEs, IATEs))
    serialize(joinpath(DESTINATION_DIR, "estimands.jls"), Configuration(estimands=estimands))
end

main()
