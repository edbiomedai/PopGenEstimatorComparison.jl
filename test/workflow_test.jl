module TestWorkflow

using Test
using JLD2
using TargetedEstimation
using TMLE
using DataFrames

@testset "Test Workflow" begin
    # For some reason if Julia is installed with juliaup on MacOS, the executable is not in ENV["PATH"]
    # Only Test the workflow runs for now
    r = run(addenv(
        `nextflow run main.nf -c test/assets/testconfig.config -resume`, 
        "PATH" => ENV["PATH"] * ":" * Sys.BINDIR
    ))
    @test r.exitcode == 0
    # Check the Aggregated Results Files 
    ## 2 estimator files: 2 top level entries in the results dict
    ## 2 sample sizes: 2 sub entries in each top level entry
    ## 5 Estimands, 1 random seed, 2 repeats = 10 lines per dataframe
    expected_unique_random_seeds_repeats = DataFrame(
        REPEAT_ID = [1, 2],
        RNG_SEED  = [0, 0]
    )
    for file in (joinpath("results", "from_densities_results.hdf5"), joinpath("results", "permutation_results.hdf5"))
        jldopen(file) do io
            results = io["results"]
            # Top level entry 1: two sub entries
            glmnet_100 = results[(:wTMLE_GLMNET, :TMLE_GLMNET, :OSE_GLMNET)][100]
            @test size(glmnet_100) == (10, 5)
            @test sort(unique(glmnet_100[!, [:REPEAT_ID, :RNG_SEED]])) == expected_unique_random_seeds_repeats
            @test count(x -> x isa TMLE.Estimate, glmnet_100.TMLE_GLMNET) >= 4

            glmnet_200 = results[(:wTMLE_GLMNET, :TMLE_GLMNET, :OSE_GLMNET)][200]
            @test size(glmnet_200) == (10, 5)
            @test sort(unique(glmnet_200[!, [:REPEAT_ID, :RNG_SEED]])) == expected_unique_random_seeds_repeats
            @test count(x -> x isa TMLE.Estimate, glmnet_200.TMLE_GLMNET) >= 4

            # Top level entry 2: two sub entries
            xgboost_100 = results[(:wTMLE_XGBOOST, :TMLE_XGBOOST, :OSE_XGBOOST)][100]
            @test size(xgboost_100) == (10, 5)
            @test sort(unique(xgboost_100[!, [:REPEAT_ID, :RNG_SEED]])) == expected_unique_random_seeds_repeats
            @test count(x -> x isa TMLE.Estimate, xgboost_100.TMLE_XGBOOST) >= 4

            xgboost_200 = results[(:wTMLE_XGBOOST, :TMLE_XGBOOST, :OSE_XGBOOST)][200]
            @test size(xgboost_200) == (10, 5)
            @test sort(unique(xgboost_200[!, [:REPEAT_ID, :RNG_SEED]])) == expected_unique_random_seeds_repeats
            @test count(x -> x isa TMLE.Estimate, xgboost_200.TMLE_XGBOOST) >= 4
        end
    end

    # Check the Analysis Output
    for file in (
        joinpath("results", "density_estimation", "analysis", "analysis1D", "summary_stats.hdf5"),
        joinpath("results", "permutation_estimation", "analysis", "analysis1D", "summary_stats.hdf5") 
        )
        results = jldopen(io -> io["results"], file)
        @test names(results) == ["ESTIMAND", "ESTIMATOR", "SAMPLE_SIZE", "BIAS", "VARIANCE", "MSE", "COVERAGE", "CI_WIDTH"]
    end
end

end