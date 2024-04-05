module TestAnalysis

using Test
using PopGenEstimatorComparison

TESTDIR = joinpath(pkgdir(PopGenEstimatorComparison), "test")


@testset "Test analysis" begin
    estimation_results_file = joinpath(TESTDIR, "assets", "estimation_results.hdf5")
    dataset_file = joinpath(TESTDIR, "assets", "dataset.arrow")
    copy!(ARGS, [

    ])
    PopGenEstimatorComparison.julia_main()
end

end

true