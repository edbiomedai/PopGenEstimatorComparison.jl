module PopGenEstimatorComparison

using Distributions
using LinearAlgebra
using Random
using TargetedEstimation
using DataFrames
using Flux
using MLJBase
using StableRNGs
using TMLE

include("generative_models.jl")
include("estimation.jl")

export RandomDatasetGenerator
export MixtureDensityNetwork, MixtureDensityNetworkEstimator
export sample, train!
export estimate_from_simulated_dataset

end
