module PopGenEstimatorComparison

using Distributions
using Random
using TargetedEstimation
using DataFrames
using Flux
using MLJBase
using TMLE
using OneHotArrays
using MLJModels
using MLJLinearModels
using MLJGLMInterface
using CategoricalArrays
using StatsBase
using JLD2
using Tables
using Arrow
using ArgParse
using JSON

include("utils.jl")

include(joinpath("density_estimation", "glm.jl"))
include(joinpath("density_estimation", "neural_net.jl"))
include(joinpath("density_estimation", "model_selection.jl"))

include(joinpath("samplers", "permutation_null_sampler.jl"))
# include(joinpath("samplers", "density_estimation_sampler.jl"))

include("estimation.jl")

include("cli.jl")

export PermutationNullSampler, DensityEstimationSampler
export MixtureDensityNetwork, CategoricalMLP
export NeuralNetworkEstimator
export GLMEstimator
export sample_from, train!, evaluation_metrics
export density_estimation, density_estimation_inputs
export estimate_from_simulated_dataset
export permutation_sampling_estimation
export read_df_results, read_df_result
export save_aggregated_df_results

end
