"""
This is a hardcoded function providing a list of density estimators
"""
function test_density_estimators(X, y; batchsize=16)
    hidden_sizes_list = ((20,), (20, 20))
    neural_nets = (NeuralNetworkEstimator(X, y; hidden_sizes=hidden_sizes, batchsize=batchsize) for hidden_sizes in hidden_sizes_list)
    glms = (GLMEstimator(X, y),)
    return collect(Iterators.flatten((neural_nets, glms)))
end

function get_density_estimators end

function get_density_estimators(file::String, X, y)
    include(abspath(file))
    return Base.invokelatest(get_density_estimators, X, y)
end

get_density_estimators(::Nothing, X, y) = test_density_estimators(X, y)

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
    estimators_list=nothing,
    output=string("density_estimate_", outcome, ".hdf5"),
    train_ratio=10,
    verbosity=1
    )
    outcome, parents = read_density_variables(density_file)
    dataset = TargetedEstimation.instantiate_dataset(dataset_file)
    TargetedEstimation.coerce_types!(dataset, [outcome, parents...])

    X, y = X_y(dataset, parents, outcome)
    density_estimators = get_density_estimators(estimators_list, X, y)
    X_train, y_train, X_test, y_test = train_validation_split(X, y; train_ratio=train_ratio)
    metrics = []
    for estimator in density_estimators
        train!(estimator, X_train, y_train, verbosity=verbosity-1)
        train_loss = evaluation_metrics(estimator, X_train, y_train).logloss
        test_loss = evaluation_metrics(estimator, X_test, y_test).logloss
        push!(metrics, (train_loss=train_loss, test_loss=test_loss))
    end
    # Retrain best
    best_estimator_id = findmin(x -> x.test_loss, metrics)[2]
    best_estimator = get_density_estimators(estimators_list, X, y)[best_estimator_id]
    train!(best_estimator, X, y, verbosity=verbosity-1)
    # Save
    if output !== nothing
        jldopen(output, "w") do io
            io["outcome"] = outcome
            io["parents"] = parents
            io["estimators"] = serializable!(density_estimators)
            io["metrics"] = metrics
            io["best-estimator"] = serializable!(best_estimator)
        end
    end
    return 0
end
