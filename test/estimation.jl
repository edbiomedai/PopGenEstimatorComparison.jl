module TestEstimation

using Test
using PopGenEstimatorComparison
using TMLE
using Random
using JLD2
using Distributions

@testset "Test estimate_from_simulated_dataset" begin
    generator = RandomDatasetGenerator(treatments_distribution=Bernoulli(), outcome_distribution=Normal())
    config = TMLE.Configuration(
        estimands=[ATE(
            outcome=:Y,
            treatment_values = (T_1 = (case=1, control=0),),
            treatment_confounders = (:W_1, :W_2, :W_3, :W_4, :W_5, :W_6)
            )]
    )
    n = 100
    estimators = (TMLE=TMLEE(), )
    estimate_from_simulated_dataset(generator, n, config, estimators;
        hdf5_output="output.hdf5",
        verbosity=0, 
        rng=MersenneTwister(123), 
        chunksize=100
    )
    jldopen("output.hdf5") do io
        results = io["Batch_1"]
        @test results[1].TMLE isa TMLE.TMLEstimate
    end
    rm("output.hdf5")
end

end

true