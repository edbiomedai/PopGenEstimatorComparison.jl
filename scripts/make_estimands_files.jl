using TMLE
using Arrow
using DataFrames
using Serialization

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
    DATASET_FILE = joinpath("results", "dataset.arrow")
    DESTINATION_DIR = joinpath("assets", "estimands")
    @assert isdir(DESTINATION_DIR)
    dataset = Arrow.Table(DATASET_FILE) |> DataFrame

    variables = variables_from_dataset(dataset)

    estimands = factorialEstimands(
        ATE, 
        dataset,
        [:rs1421085],
        variables.outcomes[1:10],
        confounders=variables.confounders,
        outcome_extra_covariates=variables.outcome_extra_covariates
    )
    serialize(joinpath(DESTINATION_DIR, "rs1421085_ATEs.jls"), Configuration(estimands=estimands))
end

main()
