function test_density_estimators(X, y; batchsize=16)
    snne = SieveNeuralNetworkEstimator(X, y; 
        hidden_sizes_candidates=[(20,), (20, 20)], 
        batchsize=batchsize
    )
    glm = GLMEstimator(X, y)
    return (snne=snne, glm=glm)
end

function study_density_estimators(X, y)
    snne = SieveNeuralNetworkEstimator(X, y; 
        hidden_sizes_candidates=[(5,), (10,), (20,), (40,), (60,), (80,), (100,), (120,), (140,)], 
        max_epochs=10_000,
        sieve_patience=5,
        batchsize=64,
        patience=5
    )
    glm = GLMEstimator(X, y)
    return (snne=snne, glm=glm)
end

function get_density_estimators(mode, X, y)
    density_estimators = if mode == "test"
        test_density_estimators(X, y)
    else
        study_density_estimators(X, y)
    end
    return density_estimators
end

function read_density_variables(file)
    d = JSON.parsefile(file)
    return Symbol(d["outcome"]), Symbol.(d["parents"])
end

function is_compatible_with_group(conditional_densities, group_conditional_densities)
    for (outcome, parents) in conditional_densities
        if haskey(group_conditional_densities, outcome)
            if group_conditional_densities[outcome] != parents
                return false
            end
        end
    end
    return true
end

"""
    make_compatible_estimands_groups(estimands)

Split estimands in compatible groups. Each group must correspond to exactly the same generating process.
Each generating process is determined by:
    - A set of propensity scores
    - An outcome model

Ince confounders are always the same PCs for every single treatment variable, the propensity scores are all compatible with each other.
Since confounders and covariates are always the same across outcome models, the outcome models are only compatible if they share exactly the same treatments.
"""
function make_compatible_estimands_groups(estimands)
    groups = []
    for Ψ in estimands
        conditional_densities = Dict(
            (f.outcome => f.parents) for f in TMLE.nuisance_functions_iterator(Ψ)
        )
        new_group_required = true
        for group in groups
            if is_compatible_with_group(conditional_densities, group.conditional_densities)
                push!(group.estimands, Ψ)
                merge!(group.conditional_densities, conditional_densities)
                new_group_required = false
                break
            end
        end
        if new_group_required
            push!(groups, (estimands=Any[Ψ], conditional_densities=conditional_densities))
        end
    end

    return groups
end

function density_estimation_inputs(datasetfile, estimands_prefix; batchsize=10, output_prefix="de_")
    estimands_dir, _prefix = splitdir(estimands_prefix)
    _estimands_dir = estimands_dir == "" ? "." : estimands_dir

    estimand_files = map(
        f -> joinpath(estimands_dir, f),
        filter(
            f -> startswith(f, _prefix), 
            readdir(_estimands_dir)
        )
    )
    dataset = TargetedEstimation.instantiate_dataset(datasetfile)
    estimands = reduce(
        vcat,
        TargetedEstimation.instantiate_estimands(f, dataset) for f in estimand_files
    )
    estimand_groups = make_compatible_estimands_groups(estimands)
    for (group_index, group) in enumerate(estimand_groups)
        group_prefix = string(output_prefix, "group_", group_index)
        # Write estimands
        for (batch_index, batch) in enumerate(Iterators.partition(group.estimands, batchsize))
            batch_filename = string(group_prefix, "_estimands_", batch_index, ".jls")
            serialize(batch_filename, TMLE.Configuration(estimands=batch))
        end
        # Write conditional densities
        for (cd_index, (outcome, parents)) in enumerate(group.conditional_densities)
            conditional_density_filename = string(group_prefix, "_conditional_density_", cd_index, ".json")
            open(conditional_density_filename, "w") do io
                JSON.print(io, Dict("outcome" => outcome, "parents" => parents), 1)
            end
        end
    end
end

serializable!(estimators::AbstractVector) = [serializable!(estimator) for estimator in estimators]

function density_estimation(
    dataset_file,
    density_file;
    mode="study",
    output=string("density_estimate.hdf5"),
    train_ratio=10,
    verbosity=1
    )
    outcome, parents = read_density_variables(density_file)
    dataset = TargetedEstimation.instantiate_dataset(dataset_file)
    TargetedEstimation.coerce_types!(dataset, [outcome, parents...])

    X, y = X_y(dataset, parents, outcome)
    density_estimators = get_density_estimators(mode, X, y)
    X_train, y_train, X_test, y_test = train_validation_split(X, y; train_ratio=train_ratio)
    metrics = []
    for estimator in density_estimators
        train!(estimator, X_train, y_train, verbosity=verbosity-1)
        train_loss = evaluation_metrics(estimator, X_train, y_train).logloss
        test_loss = evaluation_metrics(estimator, X_test, y_test).logloss
        push!(metrics, (train_loss=train_loss, test_loss=test_loss))
    end
    # Retrain Sieve Neural Network
    snne = get_density_estimators(mode, X, y).snne
    train!(snne, X, y, verbosity=verbosity-1)
    # Save
    if output !== nothing
        jldopen(output, "w") do io
            io["outcome"] = outcome
            io["parents"] = parents
            io["estimators"] = PopGenEstimatorComparison.serializable!(density_estimators)
            io["metrics"] = metrics
            io["sieve-neural-net"] = PopGenEstimatorComparison.serializable!(snne)
        end
    end
    return 0
end
