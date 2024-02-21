module TestGenerativeModels

using PopGenEstimatorComparison
using Test
using Random
using DataFrames

@testset "Test RandomDatasetGenerator" begin
    generator = RandomDatasetGenerator()
    sample_size = 10
    dataset = sample(generator, sample_size)
    @test names(dataset) == ["W_1", "W_2", "W_3", "W_4", "W_5", "W_6", "T_1", "Y"]
    @test size(dataset) == (sample_size, length(names(dataset)))
end

end

true