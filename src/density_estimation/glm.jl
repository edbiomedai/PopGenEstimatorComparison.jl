mutable struct GLMEstimator
    model::Union{MLJLinearModels.LogisticClassifier, MLJGLMInterface.LinearRegressor}
    machine
    GLMEstimator(model) = new(model)
end

GLMEstimator(X, y::CategoricalVector) = GLMEstimator(MLJLinearModels.LogisticClassifier()) 
GLMEstimator(X, y) = GLMEstimator(MLJGLMInterface.LinearRegressor()) 

function train!(estimator::GLMEstimator, X, y; verbosity=1)
    mach = machine(estimator.model, X, y)
    fit!(mach, verbosity=verbosity)
    estimator.machine = mach
    return estimator
end