get_sampler(::Nothing, estimands) = PermutationSampler(estimands)

get_sampler(prefix::AbstractString, estimands) =
    DensityEstimateSampler(prefix, estimands)

function estimate_from_simulated_data(
    origin_dataset, 
    estimands_config, 
    estimators_config;
    sample_size=nothing,
    sampler_config=nothing,
    nrepeats=10,
    out="output.arrow",
    verbosity=1,
    rng_seed=0, 
    chunksize=100,
    workdir=mktempdir()
    )
    rng = Random.default_rng()
    Random.seed!(rng, rng_seed)
    origin_dataset = TargetedEstimation.instantiate_dataset(origin_dataset)
    sample_size = sample_size !== nothing ? sample_size : nrow(origin_dataset)
    estimands = TargetedEstimation.instantiate_estimands(estimands_config, origin_dataset)
    all_variables = collect(union((TargetedEstimation.variables(arg) for arg in estimands)...))
    TargetedEstimation.coerce_types!(origin_dataset, all_variables)
    estimators_spec = TargetedEstimation.instantiate_estimators(estimators_config)
    sampler = get_sampler(sampler_config, estimands)
    statistics = []
    for repeat_id in 1:nrepeats
        outfilename = repeat_filename(workdir, repeat_id)
        outputs = TargetedEstimation.Outputs(hdf5=TargetedEstimation.HDF5Output(filename=outfilename))
        sampled_dataset = sample_from(sampler, origin_dataset; n=sample_size)
        append!(statistics, compute_statistics(sampled_dataset, estimands))
        runner = Runner(
            sampled_dataset;
            estimands_config=estimands_config, 
            estimators_spec=estimators_spec, 
            verbosity=verbosity, 
            outputs=outputs, 
            chunksize=chunksize,
            cache_strategy="release-unusable",
            sort_estimands=false
        )
        runner()
    end

    results = PopGenEstimatorComparison.read_results_dir(workdir)
    results.SAMPLE_SIZE .= sample_size
    results.RNG_SEED .= rng_seed
    results.STATISTICS .= statistics

    jldsave(out, results=results, statistics_by_repeat_id=statistics)
end
