module PopGenEstimatorComparison

using MLJ
using MLJBase
using GLMNet
using CSV
using DataFrames
using CairoMakie
using Arrow
using PopGenEstimatorComparison
using MLJLinearModels
using TMLE
using PopGenEstimatorComparison
using MLJXGBoostInterface
import MLJGLMInterface as MLJGLM

export GLMNetClassifier, GLMNetRegressor
export compare_estimators

include("glmnet.jl")
include("comparison.jl")

end
