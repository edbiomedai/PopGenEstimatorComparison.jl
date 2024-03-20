function get_density_estimators(X, y)
    hidden_sizes_list = ((20,), (20, 20))
    neural_nets = (NeuralNetworkEstimator(X, y; batchsize=16, hidden_sizes=hidden_sizes) for hidden_sizes in hidden_sizes_list)
    glms = (GLMEstimator(X, y),)
    return collect(Iterators.flatten((neural_nets, glms)))
end