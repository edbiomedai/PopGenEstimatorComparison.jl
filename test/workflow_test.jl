module TestWorkflow

using Test
using JLD2
using TargetedEstimation
using TMLE

@testset "Test Workflow" begin
    # For some reason if Julia is installed with juliaup on MacOS, the executable is not in ENV["PATH"]
    # Only Test the workflow runs for now
    r = run(addenv(
        `nextflow run main.nf -c test/assets/testconfig.config -resume`, 
        "PATH" => ENV["PATH"] * ":" * Sys.BINDIR
    ))
    @test r.exitcode == 0

    jldopen(joinpath("results", "permutation_results.hdf5")) do io
        results = io["results"]
        @test size(results) == (16, 4)
        @test Set(results.REPEAT_ID) == Set([1, 2])
        @test Set(results.SAMPLE_SIZE) == Set([100, 200])
        @test Set(results.RNG_SEED) == Set([0])
        # non regression
        @test count(x -> x isa TMLE.Estimate, results.TMLE) > 5
    end

    jldopen(joinpath("results", "from_densities_results.hdf5")) do io
        results = io["results"]
        @test size(results) == (16, 4)
        @test Set(results.REPEAT_ID) == Set([1, 2])
        @test Set(results.SAMPLE_SIZE) == Set([100, 200])
        @test Set(results.RNG_SEED) == Set([0])
        # non regression
        @test count(x -> x isa TMLE.Estimate, results.TMLE) > 5
    end
end

end