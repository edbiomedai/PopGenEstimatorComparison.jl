mutable struct GLMEstimator
    model::Union{MLJGLM.LinearBinaryClassifier, MLJGLM.LinearRegressor}
end

make_categorical_or_float(vector) =
    MLJBase.autotype(vector) <: Union{Missing, <:Finite} ? 
    categorical(vector) : 
    float(vector)

function (estimator::LinearEstimator)(Ψ, dataset; verbosity=1)
    features = [keys(Ψ.treatment_values)..., first(Ψ.treatment_confounders)..., Ψ.outcome_extra_covariates...]
    outcome = Ψ.outcome
    if Ψ isa TMLE.StatisticalATE
        if length(Ψ.treatment_values) > 1
            # Conditional ATE, dataset is filtered
            eQTL_values = values(Ψ.treatment_values)[2]
            if eQTL_values.case == eQTL_values.control
                eqtl = keys(Ψ.treatment_values)[2]
                train_dataset = dropmissing(filter(x -> x[eqtl] == eQTL_values.case, dataset[!, vcat(outcome, features)]))
                features = filter(x -> x !== eqtl, features)
            else
                throw(ArgumentError(string("Don't know how to perform linear inference with: ", Ψ)))
            end
        else
            train_dataset = dataset
        end
    else
        throw(ArgumentError(string("Don't know how to perform linear inference with: ", Ψ)))
    end

    X = train_dataset[!, features]
    for colname in names(X)
        X[!, colname] = make_categorical_or_float(train_dataset[!, colname])
    end
    y = make_categorical_or_float(train_dataset[!, Ψ.outcome])
    model = OneHotEncoder() |> estimator.model
    mach = machine(model, X, y)
    fit!(mach, verbosity=verbosity)
    coef_table = report(mach).linear_regressor.coef_table
    return coef_table
end