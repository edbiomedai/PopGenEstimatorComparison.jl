function get_density_estimators(X, y)
    snne = SieveNeuralNetworkEstimator(X, y; 
        hidden_sizes_candidates=[(20,), (40,), (60,), (80,), (100,), (120,), (140,)], 
        max_epochs=10_000,
        sieve_patience=3,
        batchsize=64,
        patience=5
    )
    glm = GLMEstimator(X, y)
    return [snne, glm]
end