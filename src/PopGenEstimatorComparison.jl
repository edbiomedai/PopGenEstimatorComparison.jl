module PopGenEstimatorComparison

using Distributions
using LinearAlgebra
using Random
using TargetedEstimation
using DataFrames

include("generative_models.jl")
include("estimation.jl")

export RandomDatasetGenerator
export sample
export estimate_from_simulated_dataset

end
