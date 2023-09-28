module PopGenEstimatorComparison

using MLJ
using MLJBase
using GLMNet
import MLJGLMInterface as MLJGLM


export GLMNetClassifier, GLMNetRegressor

include("glmnet.jl")

end
