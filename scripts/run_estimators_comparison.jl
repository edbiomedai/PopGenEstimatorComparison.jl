using ArgParse

function parse_commandline()
    s = ArgParseSettings(
        description = "Comparing Semi-Parametric estimators on the UKB.",
        version = "0.1",
        add_version = true)

    @add_arg_table s begin
        "--id"
            help = "Estimand index to run the analysis for."
            arg_type = Int
            default = 1
        "--config-file"
            help = "Path to config file"
            default = "config/config.jl"
            arg_type = String
        "--out", "-o"
            help = "Verbosity level"
            arg_type = String
            default = "estimators_comparison.png"
        "--verbosity", "-v"
            help = "Verbosity level"
            arg_type = Int
            default = 1
    end

    return parse_args(s)
end

parsed_args = parse_commandline()

using PopGenEstimatorComparison

compare_estimators(parsed_args)