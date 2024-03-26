module TestDensityEstimation

###########################################################
###        DENSITY_ESTIMATION WORKFLOW TESTS        ###
###########################################################
# Details
# -------
# This module tests the DENSITY_ESTIMATION Workflow
# This workflow is itself composed of four steps:
#   - density estimation inputs generation: conditional densities to be estimated and target estimands
#   - density estimation
#   - estimation from density estimates
#   - aggregate results
#
# We first test each script command independently, then make sure the Nextflow workflow runs

using Test
using PopGenEstimatorComparison
using TMLE
using Random
using JLD2
using Distributions
using LogExpFunctions
using DataFrames
using JLD2
using JSON

PKGDIR = pkgdir(PopGenEstimatorComparison)

TESTDIR = joinpath(PKGDIR, "test")

include(joinpath(TESTDIR, "testutils.jl"))

@testset "Integration Test" begin
    dataset_file = joinpath(TESTDIR, "assets", "dataset.arrow")
    estimands_prefix = joinpath(TESTDIR, "assets", "estimands", "estimands_ate")
    # Inputs generation
    de_inputs_dir = mktempdir()
    copy!(ARGS, [
        "density-estimation-inputs",
        dataset_file,
        estimands_prefix,
        "--batchsize=2",
        string("--output-prefix=", joinpath(de_inputs_dir, "de_"))
    ]
    )
    PopGenEstimatorComparison.julia_main()
    # Density Estimation
    density_estimators_file = joinpath(TESTDIR, "assets", "density_estimators.jl")
    density_estimates_dir = mktempdir()
    for density_file in readdir(de_inputs_dir)
        if endswith(density_file, "json")
            file_splits = split(density_file, "_")
            file_id = file_splits[end][1:end-4]
            outfile = joinpath(
                density_estimates_dir,
                string(join(file_splits[1:end-1], "_"), "_estimate_", file_id, "hdf5")
                )
            copy!(ARGS, [
                "density-estimation",
                dataset_file,
                joinpath(de_inputs_dir, density_file),
                string("--estimators=", density_estimators_file),
                string("--output=", outfile),
                "--train-ratio=7",
                "--verbosity=0"
            ])
            PopGenEstimatorComparison.julia_main()
            @test !(PopGenEstimatorComparison.best_density_estimator(outfile) isa Nothing)
            density_dict = JSON.parsefile(joinpath(de_inputs_dir, density_file))
            jldopen(outfile) do io
                @test io["outcome"] == density_dict["outcome"]
                @test io["parents"] == density_dict["parents"]
                @test all(haskey(v, :train_loss) for v in io["metrics"])
                @test all(haskey(v, :test_loss) for v in io["metrics"])
            end
        end
    end
    # Estimation From Densities
    estimators_config = joinpath(PKGDIR, "assets", "estimators-configs", "glm.jl")
    estimands_file = joinpath(de_inputs_dir, "de_group_1_estimands_1.jls")
    outdir = mktempdir()
    copy!(ARGS, [
        "estimation",
        dataset_file,
        estimands_file,
        estimators_config,
        "--sample-size=100",
        string("--density-estimates-prefix=", joinpath(density_estimates_dir, "de")),
        "--n-repeats=1",
        string("--out=", joinpath(outdir, "results.hdf5")),
        "--verbosity=0",
        "--rng=1",
        "--chunksize=100",
        string("--workdir=", outdir)
    ])
    PopGenEstimatorComparison.julia_main()

    jldopen(joinpath(outdir, "results.hdf5")) do io
        results = io["results"]
        @test results.REPEAT_ID == [1, 1]
        @test results.RNG_SEED == [1, 1]
        @test results.SAMPLE_SIZE == [100, 100]
        # At least one success
        @test count(x -> x isa TMLE.Estimate, results.TMLE) > 0
    end
end

end

true