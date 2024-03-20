"""
This is a hardcoded function providing a list of density estimators
"""
function test_density_estimators(X, y)
    hidden_sizes_list = ((20,), (20, 20))
    neural_nets = (NeuralNetworkEstimator(X, y; hidden_sizes=hidden_sizes) for hidden_sizes in hidden_sizes_list)
    glms = (GLMEstimator(X, y),)
    return collect(Iterators.flatten((neural_nets, glms)))
end

function get_density_estimators end

function get_density_estimators(file::String, X, y)
    include(abspath(file))
    return Base.invokelatest(get_density_estimators, X, y)
end

function read_density_variables(file)
    d = JSON.parsefile(file)
    return d["outcome"], d["parents"]
end

best_density_estimator(estimators, metrics) = estimators[findmin(x -> x.test_loss, metrics)[2]]

function coerce_types!(dataset, colnames)
    for colname in colnames
        if autotype(dataset[!, colname]) <: Finite
            dataset[!, colname] = categorical(dataset[!, colname])
        else
            dataset[!, colname] = float(dataset[!, colname])
        end
    end
end

function density_estimation_inputs(datasetfile, estimands_prefix; output_prefix="conditional_density_variables_")
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
    conditional_densities_variables = get_conditional_densities_variables(estimands)
    for (index, conditional_density_variables) ∈ enumerate(conditional_densities_variables)
        open(string(output_prefix, index, ".json"), "w") do io
            JSON.print(io, conditional_density_variables, 1)
        end
    end
end

function density_estimation(
    dataset_file,
    density_file;
    estimators_list=nothing,
    output=string("density_estimate_", outcome, ".hdf5"),
    train_ratio=10,
    verbosity=1
    )
    outcome, features = read_density_variables(density_file)
    dataset = TargetedEstimation.instantiate_dataset(dataset_file)
    coerce_types!(dataset, [outcome, features...])
    
    X, y = X_y(dataset, features, outcome)
    density_estimators = get_density_estimators(estimators_list, X, y)
    X_train, y_train, X_test, y_test = train_validation_split(X, y; train_ratio=train_ratio)
    metrics = []
    for estimator in density_estimators
        train!(estimator, X_train, y_train, verbosity=verbosity-1)
        train_loss = evaluation_metrics(estimator, X_train, y_train).logloss
        test_loss = evaluation_metrics(estimator, X_test, y_test).logloss
        push!(metrics, (train_loss=train_loss, test_loss=test_loss))
    end
    if output !== nothing
        jldopen(output, "w") do io
            io["outcome"] = outcome
            io["features"] = features
            io["estimators"] = density_estimators
            io["metrics"] = metrics
        end
    end
    return 0
end