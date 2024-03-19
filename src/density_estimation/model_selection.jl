evaluate(estimator, X_val, y_val) = 0

function get_density_estimators(X, y)
    hidden_sizes_list = ((20,), (20, 20))
    neural_nets = (NeuralNetworkEstimator(X, y, hidden_sizes) for hidden_sizes in hidden_sizes_list)

    return collect(neural_nets)
end

function evaluate_and_save_density_estimators!(
    dataset,
    outcome, 
    features;
    outpath=nothing,
    train_ratio=10,
    verbosity=1
    )
    X, y = X_y(dataset, features, outcome)
    density_estimators = get_density_estimators(X, y)
    X_train, y_train, X_val, y_val = train_validation_split(X, y; train_ratio=train_ratio)
    metrics = []
    for estimator in density_estimators
        train!(estimator, X_train, y_train, verbosity=verbosity-1)
        push!(metrics, evaluate(estimator, X_val, y_val))
    end
    if outpath !== nothing
        jldopen(outpath, "w") do io
            io["estimators"] = density_estimators
            io["metrics"] = metrics
        end
    end
    return density_estimators, metrics
end
