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
using Serialization
using Makie
using CSV
using CairoMakie

include("utils.jl")

include(joinpath("density_estimation", "glm.jl"))
include(joinpath("density_estimation", "neural_net.jl"))
include(joinpath("density_estimation", "inputs_from_gene_atlas.jl"))
include(joinpath("density_estimation", "model_selection.jl"))

include(joinpath("samplers", "permutation_null_sampler.jl"))
include(joinpath("samplers", "density_estimate_sampler.jl"))

include("estimands.jl")
include("estimation.jl")
include("analysis.jl")
include("cli.jl")

export SaveVariantsAndEstimands
export PermutationSampler, DensityEstimateSampler
export theoretical_true_effect, empirical_true_effect, true_effect
export MixtureDensityNetwork, CategoricalMLP
export NeuralNetworkEstimator, SieveNeuralNetworkEstimator
export GLMEstimator
export sample_from, train!, evaluation_metrics
export density_estimation, density_estimation_inputs
export density_estimation_inputs_from_gene_atlas
export estimate_from_simulated_data
export save_aggregated_df_results
export analyse

end
