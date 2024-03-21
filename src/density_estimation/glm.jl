mutable struct GLMEstimator
    model
    machine
    GLMEstimator(model) = new(model)
end

GLMEstimator(X, y::CategoricalVector) = 
    GLMEstimator(OneHotEncoder(drop_last=true) |> MLJLinearModels.LogisticClassifier()) 
GLMEstimator(X, y) = 
    GLMEstimator(OneHotEncoder(drop_last=true) |> MLJGLMInterface.LinearRegressor()) 

function train!(estimator::GLMEstimator, X, y; verbosity=1)
    mach = machine(estimator.model, X, y, cache=false)
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

sample_from(estimator::GLMEstimator, X, labels) = sample_from(estimator::GLMEstimator, X)

function evaluation_metrics(estimator::GLMEstimator, X, y)
    ŷ = MLJBase.predict(estimator.machine, X)
    logloss = -mean(logpdf.(ŷ, y))
    return (logloss = logloss,)
end