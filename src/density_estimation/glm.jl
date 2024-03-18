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

"""
For each row in X, samples a new y
"""
function sample_from(estimator::GLMEstimator, X)
    ŷ = MLJBase.predict(estimator.machine, X)
    return rand.(ŷ)
end

function evaluation_metrics(estimator::GLMEstimator, X, y)
    ŷ = MLJBase.predict(estimator.machine, X)
    logloss = mean(logpdf.(ŷ, y))
    return (logloss = logloss,)
end