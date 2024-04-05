module TestUtils

using PopGenEstimatorComparison
using PopGenEstimatorComparison: get_input_size, getlabels,
    train_validation_split, net_train_validation_split,
    get_outcome, confounders_and_covariates_set, get_treatments
using Test
using CategoricalArrays
using DataFrames
using MLJBase
using TMLE
using DataFrames
using Arrow

TESTDIR = joinpath(pkgdir(PopGenEstimatorComparison), "test")

include(joinpath(TESTDIR, "testutils.jl"))

@testset "Test compute_statistics" begin
    dataset = DataFrame(Arrow.Table(joinpath(TESTDIR, "assets", "dataset.arrow")))
    estimands = linear_interaction_dataset_ATEs().estimands
    statistics = PopGenEstimatorComparison.compute_statistics(dataset, estimands)
    # Continuous outcome, one treatment
    @test names(statistics[1][:T₁]) == ["T₁", "proprow", "nrow"]
    # Binary outcome, two treatments
    for key ∈ (:Ybin, :T₁, :T₂, (:T₁, :T₂) ,(:Ybin, :T₁, :T₂))
        stats = statistics[2][key]
        @test stats isa DataFrame
        @test hasproperty(stats, "proprow")
        @test hasproperty(stats, "nrow")
    end
end

@testset "Test estimands variables accessors" begin
    Ψ, composedΨ = linear_interaction_dataset_ATEs().estimands

    @test confounders_and_covariates_set(Ψ) == Set([:W, :C])
    @test get_outcome(Ψ) == :Ycont
    @test get_treatments(Ψ) == (:T₁,)

    @test confounders_and_covariates_set(composedΨ) == Set([:W, :C])
    @test get_outcome(composedΨ) == :Ybin
    @test get_treatments(composedΨ) == (:T₁, :T₂)
end

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
    X_train, y_train, X_val, y_val = train_validation_split(X, y)
    @test size(y_train, 1) == 90
    @test size(y_val, 1) == 10
    @test X_train isa NamedTuple
    @test X_val isa NamedTuple

    X_train, y_train, X_val, y_val = net_train_validation_split(X, y)
    @test size(X_train) == (2, 90)
    @test size(X_val) == (2, 10)
    @test size(y_train) == (3, 90)
    @test size(y_val) == (3, 10)
end

end

true