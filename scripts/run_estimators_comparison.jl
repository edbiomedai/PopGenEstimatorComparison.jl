using ArgParse

function parse_commandline()
    s = ArgParseSettings(
        description = "Comparing Semi-Parametric estimators on the UKB.",
        version = "0.1",
        add_version = true)

    @add_arg_table s begin
        "dataset"
            help = "Full dataset"
            arg_type = String
        "estimand-list"
            help = """Summary file output by TarGene containing a list of 
            estimands that lead to significantly different results."""
            arg_type = String
        "id"
            help = "Estimand index to run the analysis for."
            arg_type = Int
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