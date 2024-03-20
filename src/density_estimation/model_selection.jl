"""
This is a hardcoded function providing a list of density estimators
"""
function test_density_estimators(X, y)
    hidden_sizes_list = ((20,), (20, 20))
    neural_nets = (NeuralNetworkEstimator(X, y; hidden_sizes=hidden_sizes) for hidden_sizes in hidden_sizes_list)
    glms = (GLMEstimator(X, y),)
    return collect(Iterators.flatten((neural_nets, glms)))
end

get_density_estimators(estimators_list::Nothing, X, y) = test_density_estimators(X, y)

function get_density_estimators(estimators_list::String, X, y)
    include(abspath(file))
    return get_density_estimators(X, y)
end

best_density_estimator(estimators, metrics) = estimators[findmin(x -> x.test_loss, metrics)[2]]

function density_estimation_inputs(datasetfile, estimands_prefix; output="conditional_densities_variables.json")
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
    open(output, "w") do io
        JSON.print(io, conditional_densities_variables, 1)
    end
end

function density_estimation(
    dataset,
    outcome, 
    features;
    estimators_list=nothing,
    output=string("density_estimate_", outcome, ".hdf5"),
    train_ratio=10,
    verbosity=1
    )
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
            io["estimators"] = density_estimators
            io["metrics"] = metrics
        end
    end
    return 0
end