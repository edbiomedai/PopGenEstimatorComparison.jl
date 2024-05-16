
function getTraits()
    return Symbol.([
        "White blood cell (leukocyte) count", 
        "sarcoidosis", 
        "D86 Sarcoidosis", 
        "G35 Multiple sclerosis", 
        "K90-K93 Other diseases of the digestive system",
        "H00-H06 Disorders of eyelid, lacrimal system and orbit", 
        "Trunk fat percentage",
        "Skin colour"
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
    # Skin colour: https://www.nature.com/articles/s41467-018-07691-z
    (:rs1805007, :rs6088372)
    (:rs1805005, :rs6059655)
    (:rs1805008, :rs1129038)
    # Parkison: https://pubmed.ncbi.nlm.nih.gov/31234232/
    (:rs1732170, :rs456998, :rs356219, :rs8111699)
    (:rs11868112, :rs6456121, :rs356219)
    # MS: https://www.sciencedirect.com/science/article/pii/S0002929723000915
    (:rs10419224, :rs59103106)
    # T2D: https://www.sciencedirect.com/science/article/pii/S0002929723000915
    (:rs9926016, :rs73323281)

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
    group_size = 5
    
    isdir(DESTINATION_DIR) || mkdir(DESTINATION_DIR)

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
    for (group_id, estimands_group) in enumerate(Iterators.partition(estimands, group_size))
        serialize(joinpath(DESTINATION_DIR, string("estimands_", group_id, ".jls")), Configuration(estimands=estimands_group))
    end
end