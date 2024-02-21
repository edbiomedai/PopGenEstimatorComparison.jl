module TestGenerativeModels

using PopGenEstimatorComparison
using Test
using Random

@testset "Test RandomDatasetGenerator" begin
    generator = RandomDatasetGenerator()
    sample_size = 10
    dataset = sample(generator, sample_size)
    @test keys(dataset) == (:Y, :T_1, :W_1, :W_2, :W_3, :W_4, :W_5, :W_6)
    @test all(size(x, 1) == sample_size for x in dataset)
end

end