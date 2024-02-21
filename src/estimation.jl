function estimate_from_simulated_dataset(generator, n::Int, config, estimators;
    hdf5_output="output.hdf5",
    verbosity=1, 
    rng=Random.default_rng(), 
    chunksize=100
    )
    outputs = TargetedEstimation.Outputs(hdf5=TargetedEstimation.HDF5Output(filename=hdf5_output))
    dataset = sample(generator, n; rng=rng)
    runner = Runner(dataset;
        estimands_config=config, 
        estimators_spec=estimators, 
        verbosity=verbosity, 
        outputs=outputs, 
        chunksize=chunksize,
        rng=rng,
        cache_strategy="release-unusable",
        sort_estimands=false
    )
    runner()
end