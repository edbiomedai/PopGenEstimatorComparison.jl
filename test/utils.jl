module TestUtils

using PopGenEstimatorComparison: get_input_size, getlabels,
    train_validation_split, net_train_validation_split
using Test
using CategoricalArrays
using DataFrames
using MLJBase

@testset "Test misc" begin
    # Test get_input_size
    ## The categorical variables counts for 2
    X = DataFrame(
        x1 = [1,2], 
        x2 = categorical([1,2])
    )
    @test get_input_size(X.x1) == 1
    @test get_input_size(X.x2) == 2
    @test get_input_size(X) == 3
    # Test getlabels
    ## Only Categorical Vectors return labels otherwise nothing
    @test getlabels(categorical(["AC", "CC", "CC"])) == ["AC", "CC"]
    @test getlabels([1, 2]) === nothing
end

@testset "Test train_validation_split" begin
    X, y = MLJBase.make_blobs()
    resampling = CV(nfolds=10)
    X_train, y_train, X_val, y_val = train_validation_split(resampling, X, y)
    @test size(y_train, 1) == 90
    @test size(y_val, 1) == 10
    @test X_train isa NamedTuple
    @test X_val isa NamedTuple

    X_train, y_train, X_val, y_val = net_train_validation_split(resampling, X, y)
    @test size(X_train) == (2, 90)
    @test size(X_val) == (2, 10)
    @test size(y_train) == (3, 90)
    @test size(y_val) == (3, 10)
end

end

true