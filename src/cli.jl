function cli_settings()
    s = ArgParseSettings(description="PopGenSim CLI.")

    @add_arg_table! s begin
        "permutation-estimation"
            action = :command
            help = "Run Estimation from Permutation Null Sampler."

        "aggregate"
            action = :command
            help = "Aggregate multiple results file created by estimation procedures"
    end

    @add_arg_table! s["aggregate"] begin
        "input-prefix"
            arg_type = String
            help = "Prefix to all files to be aggregated."
        "out"
            arg_type = String
            help = "Output path."
    end

    @add_arg_table! s["permutation-estimation"] begin
        "origin-dataset"
            arg_type = String
            help = "Path to the dataset (either .csv or .arrow)"

        "estimands-config"
            arg_type = String
            help = "A string (`factorialATE`) or a serialized TMLE.Configuration (accepted formats: .json | .yaml | .jls)"

        "estimators-config"
            arg_type = String
            help = "A julia file containing the estimators to use."
        
        "--sample-size"
            arg_type = Int
            help = "Size of simulated dataset."
            default = 100

        "--n-repeats"
            arg_type = Int
            help = "Number of simulations to run."
            default = 10

        "--out"
            arg_type = String
            default = "permutation_estimation_results.hdf5"
            help = "Output file."
        
        "--verbosity"
            arg_type = Int
            default = 0
            help = "Verbosity level"

        "--chunksize"
            arg_type = Int
            help = "Results are written in batches of size chunksize."
            default = 100

        "--rng"
            arg_type = Int
            help = "Random seed (Only used for estimands ordering at the moment)."
            default = 123
    end

    return s
end

function julia_main()::Cint
    settings = parse_args(ARGS, cli_settings())
    cmd = settings["%COMMAND%"]
    cmd_settings = settings[cmd]

    if cmd == "permutation-estimation"
        permutation_sampling_estimation(
            cmd_settings["origin-dataset"],
            cmd_settings["estimands-config"],
            cmd_settings["estimators-config"], 
            cmd_settings["sample-size"];
            nrepeats=cmd_settings["n-repeats"],
            out=cmd_settings["out"],
            verbosity=cmd_settings["verbosity"],
            rng_seed=cmd_settings["rng"], 
            chunksize=cmd_settings["chunksize"])
    elseif cmd == "aggregate"
        save_aggregated_df_results(cmd_settings["input-prefix"], cmd_settings["out"])
    end

    return 0
end