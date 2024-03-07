repeat_filename(outdir, repeat) = joinpath(outdir, string("output_", repeat, ".hdf5"))
info_filename(outdir) = joinpath(outdir, "info.toml")

function read_results_dir(outdir)
    infofile = info_filename(outdir)
    infodict = TOML.parsefile(joinpath(outdir, "info.toml"))
    sample_size = infodict["sample_size"]
    rng_seed = infodict["rng_seed"]
    results = []
    for filename in filter(!=(infofile), readdir(outdir, join=true))
        repeat_id = parse(Int, split(replace(filename, ".hdf5" => ""), "_")[end])
        fileresults = read_results_file(joinpath(outdir, filename))
        fileresults = [merge(result, (REPEAT_ID=repeat_id, SAMPLE_SIZE=sample_size,RNG_SEED=rng_seed)) for result in fileresults]
        append!(results, fileresults)
    end
    
    return DataFrame(results)
end

read_results_dirs(outdirs...) = reduce(vcat, read_results_dir(outdir) for outdir in outdirs)

function permutation_sampling_estimation(origin_dataset, estimands_config, estimators, sample_size;
    nrepeats=10,
    outdir="outputs",
    verbosity=1, 
    rng_seed=0, 
    chunksize=100
    )
    rng = Random.default_rng()
    Random.seed!(rng, rng_seed)
    sampler = PermutationNullSampler(estimands_config.estimands)
    isdir(outdir) || mkdir(outdir)
    for repeat_id in 1:nrepeats
        outfilename = repeat_filename(outdir, repeat_id)
        outputs = TargetedEstimation.Outputs(hdf5=TargetedEstimation.HDF5Output(filename=outfilename))
        sampled_dataset = sample_from(sampler, origin_dataset; n=sample_size)
        runner = Runner(sampled_dataset;
            estimands_config=estimands_config, 
            estimators_spec=estimators, 
            verbosity=verbosity, 
            outputs=outputs, 
            chunksize=chunksize,
            cache_strategy="release-unusable",
            sort_estimands=false
        )
        runner()
    end
    open(joinpath(outdir, "info.toml"), "w") do io
        TOML.print(io, Dict(
            "sample_size" => sample_size,
            "rng_seed" => rng_seed
        ))
    end
end