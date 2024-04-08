using TMLE
using Arrow
using DataFrames
using Serialization

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

function variables_from_dataset(dataset)
    confounders = Set([])
    outcome_extra_covariates = Set(["Genetic-Sex", "Age-Assessment"])
    outcomes = Set([])
    variants = Set([])
    for colname in names(dataset)
        if startswith(colname, r"PC[0-9]*")
            push!(confounders, colname)
        elseif startswith(colname, r"rs[0-9]*")
            push!(variants, colname)
        elseif colname ∈ outcome_extra_covariates
            continue
        else
            push!(outcomes, colname)
        end
    end
    variables = (
        outcomes = collect(outcomes), 
        variants = collect(variants), 
        confounders = collect(confounders), 
        outcome_extra_covariates = collect(outcome_extra_covariates)
    )
    return variables
end

function main()
    DATASET_FILE = joinpath("dataset", "results", "dataset.arrow")
    DESTINATION_DIR = joinpath("assets", "estimands")
    positivity_constraint = 0.
    @assert isdir(DESTINATION_DIR)
    dataset = Arrow.Table(DATASET_FILE) |> DataFrame

    variables = variables_from_dataset(dataset)

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
