function get_density_estimators(X, y)
    hidden_sizes_list = ((64,), (64, 64))
    neural_nets = (NeuralNetworkEstimator(X, y; hidden_sizes=hidden_sizes, max_epochs=1000, patience=5) for hidden_sizes in hidden_sizes_list)
    glms = (GLMEstimator(X, y),)
    return collect(Iterators.flatten((neural_nets, glms)))
end